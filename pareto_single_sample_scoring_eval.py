
"""Score and evaluate one sample for the single-sample workflow.

This script is launched by `pareto_single_sample_analysis.py` as a subprocess.
It computes edge scores for one selected sample and writes the temporary score
array and evaluation dataframe.
"""

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from ceap.graph import Graph
from pareto_dev_utils import *
from ceap.attribute import attribute


parser = ArgumentParser(description="Score and evaluate one sample for the single-sample workflow.")
parser.add_argument("--model", type=str, default="gpt2", help="Model name or path.")
parser.add_argument("--task", type=str, default="ioi", help="Evaluation task name.")
parser.add_argument("--metric", type=str, default="logit_diff", help="Primary metric used for evaluation and, unless --div-for-scoring is set, edge scoring.")
parser.add_argument("--evaluating-batch-size", type=int, default=256, dest="evaluating_batch_size", help="Batch size used for template dataloaders.")
parser.add_argument("--scoring-sample-number", type=int, default=256, dest="scoring_sample_number", help="Number of samples included in the single-sample workflow.")
parser.add_argument("--div-for-scoring", action="store_true", help="Use KL divergence instead of the primary metric for edge scoring.")
parser.add_argument("--ig-style", type=str, default="ceap", choices=["ceap", "eap", "eap-ig"], help="Integrated gradients variant.")
parser.add_argument("--integrated-steps", type=int, default=25, dest="integrated_steps", help="Number of IG steps (ignored for --ig-style eap).")
parser.add_argument("--max-n-data", type=int, default=1000, dest="max_n_data", help="Maximum number of datapoints to load from the dataset.")
parser.add_argument("--seed", type=int, default=123, help="Random seed for train/test split.")
parser.add_argument("--operation-id", type=int, default=0, help="Index of the single-sample dataloader to process.")
parser.add_argument("--result-folder", type=str, default="results", dest="result_folder", help="Root directory for temporary score/dataframe outputs.")
parser.add_argument("--continue", action="store_true", dest="continue_run", help="Skip this sample if both temporary outputs already exist.")
parser.add_argument(
    "--circuit-selection-strategy",
    type=str,
    default="both",
    choices=["greedy", "topn", "both"],
    dest="circuit_selection_strategy",
    help="Circuit selection strategy to evaluate.",
)
args = parser.parse_args()

result_dir = build_result_dir(args)
score_output_path = Path(result_dir) / f"edge_scores_temp_{args.operation_id}.npz"
dataframe_output_path = Path(result_dir) / f"{args.operation_id}_eval_df.pkl"
if args.continue_run and score_output_path.exists() and dataframe_output_path.exists():
    print(
        f"Skipping operation-id {args.operation_id}: found existing "
        f"{score_output_path.name} and {dataframe_output_path.name}."
    )
    raise SystemExit(0)

model = load_model(args.model)
graph = Graph.from_model(model)

normalize_ig_style(args)

task_metric, kl_div = get_task_metrics(args.metric, args.task, model)
metrics_dict = {
    "metric": partial(task_metric, reduction="none", loss=False),
}
metrics = list(metrics_dict.values())

n_edges_target_ls = get_n_edges_target_list(
    model=args.model,
    task=args.task,
    edge_number=len(graph.edges),
)

sample_train_dataloader_ls,_,_,_,_, = prepare_dataloaders_for_different_templates(
    task=args.task,
    model_name=args.model,
    max_n_data=args.max_n_data,
    scoring_sample_number=args.scoring_sample_number,
    evaluating_batch_size=args.evaluating_batch_size,
    seed=args.seed,
    return_sample_dataloader=True,
)

if not (0 <= args.operation_id < len(sample_train_dataloader_ls)):
    raise ValueError(f"operation-id {args.operation_id} is out of range for {len(sample_train_dataloader_ls)} dataloaders.")

score_dataloader = sample_train_dataloader_ls[args.operation_id]

print(f"Scoring model No.{args.operation_id}.")
attribute(
    model,
    graph,
    score_dataloader,
    partial(task_metric if not args.div_for_scoring else kl_div, reduction="sum", loss=False),
    ig_style=args.ig_style,
    integration_steps=args.integrated_steps,
)

scores = graph.scores(sort=False).detach().cpu().numpy()
np.savez_compressed(score_output_path, edge_scores=scores)

train_baselines, train_corrupted_baselines = evaluate_baseline_clean_corrupted(
    model=model,
    dataloader=score_dataloader,
    metrics=metrics,
)

strategy_dataframes = {}
for strategy in selected_circuit_strategies(args.circuit_selection_strategy):
    print(f"Edge selection with {strategy}...")
    strategy_dataframes[strategy] = circuit_selection_data_collection_per_sample(
        n_edges_target_ls=n_edges_target_ls,
        graph=graph,
        model=model,
        train_dataloader=score_dataloader,
        metrics_dict=metrics_dict,
        train_baselines=train_baselines,
        train_corrupted_baselines=train_corrupted_baselines,
        strategy=strategy,
    )
all_info_df_per_sample = merge_strategy_dataframes(strategy_dataframes)
all_info_df_per_sample.to_pickle(dataframe_output_path)
