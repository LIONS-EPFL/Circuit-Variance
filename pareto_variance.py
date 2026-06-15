"""Run template-wise circuit-size vs faithfulness tradeoff experiments.

For each task template, this script scores graph edges by averaging over samples
from that template, evaluates greedy and/or top-n circuits at several edge-count
targets, and saves the resulting per-sample faithfulness dataframe.
"""

import os
import time
from functools import partial

from tqdm import tqdm

from ceap.attribute import attribute
from ceap.graph import Graph
import argparse

from pareto_dev_utils import (
    build_result_dir,
    evaluate_baseline_clean_corrupted,
    get_n_edges_target_list,
    get_task_metrics,
    load_model,
    merge_strategy_dataframes,
    normalize_ig_style,
    prepare_dataloaders_for_different_templates,
    circuit_selection_data_collection_per_sample,
    selected_circuit_strategies,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="For resampling variance experiments.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path for the HookedTransformer.")
    parser.add_argument("--task", type=str, default="sva", help="Evaluation task name.")
    parser.add_argument("--metric", type=str, default="prob_diff", help="Primary metric used for evaluation and, unless --div-for-scoring is set, edge scoring.")
    parser.add_argument("--batch-size", type=int, default=256, dest="batch_size", help="Batch size for dataloaders.")
    parser.add_argument("--div-for-scoring", action="store_true", help="Use KL divergence for scoring.")
    parser.add_argument("--scoring-sample-number", type=int, default=1000, dest="scoring_sample_number", help="Number of samples used for scoring.")
    parser.add_argument("--ig-style", type=str, default="ceap", choices=["ceap", "eap", "eap-ig"], help="Integrated gradients variant.")
    parser.add_argument("--integrated-steps", type=int, default=20, dest="integrated_steps", help="Number of IG steps (ignored for --ig-style eap).")
    parser.add_argument("--max-n-data", type=int, default=2000, dest="max_n_data", help="Maximum number of datapoints to load from the dataset.")
    parser.add_argument("--result-folder", type=str, default="results_resampling_variance", dest="result_folder", help="Root directory for saving results.")
    parser.add_argument("--seed", type=int, default=4, help="Random seed for train/test split.")
    parser.add_argument(
        "--circuit-selection-strategy",
        type=str,
        default="greedy",
        choices=["greedy", "topn", "both"],
        dest="circuit_selection_strategy",
        help="Circuit selection strategy to evaluate.",
    )
    parser.add_argument("--continue", action="store_true", dest="continue_run", help="Skip templates whose output file already exists; assumes the same data/configuration and strategy.")
    return parser.parse_args()

args = parse_args()
script_start_time = time.time()
normalize_ig_style(args)

print("Preparing dataloaders...")
train_dataloader_ls, _, template_label_ls = prepare_dataloaders_for_different_templates(
    task=args.task, model_name=args.model, max_n_data=args.max_n_data,\
    scoring_sample_number=args.scoring_sample_number, evaluating_batch_size=args.batch_size,     
    seed=args.seed, return_df=False,
)

result_dir = build_result_dir(args,)
if args.continue_run:
    print(
        "WARNING: Continue mode assumes the same dataset/configuration as the original run "
        "(especially --max-n-data, --scoring-sample-number, --seed, and source CSV contents). "
        "If these changed, template_label ordering/membership may differ and resume may be incorrect. "
        "It also assumes the same --circuit-selection-strategy, since existing output files are reused "
        "regardless of whether they contain greedy, topn, or both strategies."
    )

for i in tqdm(range(len(train_dataloader_ls))):
    train_dataloader = train_dataloader_ls[i]
    template_label = template_label_ls[i]
    template_result_dir = os.path.join(result_dir, f"{template_label}")
    edge_score_path = os.path.join(template_result_dir, "all_averaging_score_eval.pkl")
    if args.continue_run and os.path.exists(edge_score_path):
        print(f"Skipping template index {i} ({template_label}): found existing {edge_score_path}.")
        continue
    if len(train_dataloader.dataset) == 0:
        print(f"Skipping template index {i} ({template_label}): empty train dataloader.")
        continue
    print(f"Processing template index {i}: {template_label}...")
    model = load_model(args.model)
    graph = Graph.from_model(model)
    os.makedirs(template_result_dir, exist_ok=True)

    task_metric, kl_div = get_task_metrics(args.metric, args.task, model)
    metrics_dict = {"metric": partial(task_metric, reduction="none", loss=False)}
    metrics = list(metrics_dict.values())
    train_baselines, train_corrupted_baselines = evaluate_baseline_clean_corrupted(model = model, dataloader=train_dataloader, metrics=metrics)

    print(f"Scoring edges with {args.ig_style}")
    attribute(model, graph, train_dataloader,
        partial(task_metric if not args.div_for_scoring else kl_div, reduction="sum", loss=False),
        ig_style=args.ig_style, integration_steps=args.integrated_steps,) 

    n_edges_target_ls = get_n_edges_target_list(
        model=args.model,
        task=args.task,
        edge_number=len(graph.edges),
    )

    strategy_dataframes = {}
    for strategy in selected_circuit_strategies(args.circuit_selection_strategy):
        print(f"Circuit discovery with {strategy}...")
        strategy_dataframes[strategy] = circuit_selection_data_collection_per_sample(
            n_edges_target_ls=n_edges_target_ls,
            graph=graph,
            model=model,
            train_dataloader=train_dataloader,
            metrics_dict=metrics_dict,
            train_baselines=train_baselines,
            train_corrupted_baselines=train_corrupted_baselines,
            strategy=strategy,
        )

    all_info_df = merge_strategy_dataframes(strategy_dataframes)
    all_info_df.to_pickle(edge_score_path)
    print(f"Saved all-averaging evaluation dataframe to {edge_score_path}")

elapsed = (time.time() - script_start_time) / 60
print(f"pareto_variance.py finished in {elapsed:.2f} minutes.")
