
"""Run the single-sample circuit scoring workflow.

This driver launches one subprocess per scoring sample, gathers the per-sample
edge scores/evaluations, and optionally runs template-averaged scoring for
comparison.
"""

import argparse
import os
import time
from pathlib import Path
from functools import partial
from ceap.attribute import attribute

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score each sample separately.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path.")
    parser.add_argument("--task", type=str, default="sva", help="Evaluation task name.")
    parser.add_argument("--scoring-sample-number", type=int, default=1000, dest="scoring_sample_number", help="Number of scoring samples to process.")
    parser.add_argument("--parallel-batch-size", type=int, default=20, dest="parallel_batch_size", help="Number of per-sample scoring subprocesses to launch at once.")
    parser.add_argument("--evaluating-batch-size", type=int, default=256, dest="evaluating_batch_size", help="Batch size used inside evaluation dataloaders.")
    parser.add_argument("--div-for-scoring", action="store_true", help="Use KL divergence instead of the primary metric for edge scoring.")
    parser.add_argument("--metric", type=str, default="prob_diff", help="Primary metric used for evaluation and, unless --div-for-scoring is set, edge scoring.")
    parser.add_argument("--ig-style", type=str, default="ceap", choices=["ceap", "eap", "eap-ig"], help="Integrated gradients variant.")
    parser.add_argument("--integrated-steps", type=int, default=20, dest="integrated_steps", help="Number of integration steps (ignored for --ig-style eap).")
    parser.add_argument("--max-n-data", type=int, default=2000, dest="max_n_data", help="Maximum number of datapoints to load.")
    parser.add_argument("--seed", type=int, default=4, help="Random seed.")
    parser.add_argument("--result-folder", type=str, default="results_rephrasing_samplewise_variance", dest="result_folder", help="Root directory for temporary per-sample outputs and final dataframes.")
    parser.add_argument(
        "--circuit-selection-strategy",
        type=str,
        default="both",
        choices=["greedy", "topn", "both"],
        dest="circuit_selection_strategy",
        help="Circuit selection strategy to evaluate.",
    )
    parser.add_argument(
        "--all-averaging",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="all_averaging",
        help="Whether to also generate the template-averaged scoring dataframe.",
    )
    parser.add_argument("--continue", action="store_true", dest="continue_run", help="Reuse existing individual/all-averaging outputs; assumes the same data/configuration and strategy.")
    return parser.parse_args()

args = parse_args()

from ceap.graph import Graph
from tqdm import tqdm
from pareto_dev_utils import (
    build_result_dir,
    evaluate_baseline_clean_corrupted,
    get_n_edges_target_list,
    circuit_selection_data_collection_per_sample,
    get_task_metrics,
    load_model,
    merge_strategy_dataframes,
    normalize_ig_style,
    prepare_dataloaders_for_different_templates,
    run_parallel_scoring_subprocesses,
    selected_circuit_strategies,
)

script_start_time = time.time()

result_dir = build_result_dir(args,)
individual_score_path = os.path.join(result_dir, "individual_score_eval.pkl")
all_averaging_score_path = os.path.join(result_dir, "all_averaging_score_eval.pkl")

if args.continue_run:
    print(
        "WARNING: Continue mode assumes the same dataset/configuration as the original run "
        "(especially --max-n-data, --scoring-sample-number, --seed, and source CSV contents). "
        "If these changed, sample/template ordering may differ and resume may be incorrect."
    )

if args.continue_run and os.path.exists(individual_score_path):
    print(f"Continue mode enabled: found existing {individual_score_path}.")
    per_sample_eval_df = pd.read_pickle(individual_score_path).reset_index(drop=True)
else:
    print(f"--- Individually scoring edges with {args.ig_style} using parallel subprocesses ---")
    script_path = Path(__file__).with_name("pareto_single_sample_scoring_eval.py")
    score_output_paths, dataframe_output_paths = run_parallel_scoring_subprocesses(
        args=args,
        script_path=script_path,
        result_dir=result_dir,
    )

    score_records = []
    for path in score_output_paths:
        scores = np.load(path)["edge_scores"]
        if os.path.exists(path):
            os.remove(path)
        score_records.append(scores)

    edge_score_array = np.stack(score_records, axis=0)
    edge_score_array_path = Path(result_dir) / "edge_score_array.npz"
    np.savez_compressed(edge_score_array_path, edge_scores=edge_score_array)
    print(f"Saved edge score array to {edge_score_array_path}")

    per_sample_records = []
    for path in dataframe_output_paths:
        per_sample_df = pd.read_pickle(path)
        if os.path.exists(path):
            os.remove(path)
        if len(per_sample_df) != 1:
            raise ValueError(f"Expected a single row per sample, got {len(per_sample_df)} rows in {path}")
        per_sample_records.append(per_sample_df.iloc[0].to_dict())

    per_sample_eval_df = pd.DataFrame.from_records(per_sample_records).reset_index(drop=True)
    per_sample_eval_df.to_pickle(individual_score_path)
    print(f"Saved the individual scoring dataframe to {individual_score_path}")

if not args.all_averaging:
    elapsed = (time.time() - script_start_time)/60
    print(f"Skipping all-averaging evaluation dataframe. Finished in {elapsed:.2f} minutes.")
    raise SystemExit(0)

if args.continue_run and os.path.exists(all_averaging_score_path):
    elapsed = (time.time() - script_start_time)/60
    print(f"Continue mode enabled: found existing {all_averaging_score_path}. Finished in {elapsed:.2f} minutes.")
    raise SystemExit(0)

print(f"--- All-averaging scoring edges with {args.ig_style} ---")
print("Loading template-aware dataloaders")
normalize_ig_style(args)

(
    template_train_dataloader_ls,
    _,
    template_label_ls,
) = prepare_dataloaders_for_different_templates(
    task=args.task,
    model_name=args.model,
    max_n_data=args.max_n_data,
    scoring_sample_number=args.scoring_sample_number,
    evaluating_batch_size=args.evaluating_batch_size,
    seed=args.seed,
)

template_info_dfs = []
for template_label, train_dataloader in tqdm(list(zip(template_label_ls, template_train_dataloader_ls)),total=len(template_label_ls)):
    if len(train_dataloader.dataset) == 0:
        print(f"Skipping template {template_label}: empty train dataloader.")
        continue
    print(f"Processing template {template_label}...")
    model = load_model(args.model)
    graph = Graph.from_model(model)
    n_edges_target_ls = get_n_edges_target_list(
        model=args.model,
        task=args.task,
        edge_number=len(graph.edges),
    )

    task_metric, kl_div = get_task_metrics(args.metric, args.task, model)
    metrics_dict = {
        "metric": partial(task_metric, reduction="none", loss=False),
    }
    metrics = list(metrics_dict.values())
    train_baselines, train_corrupted_baselines = evaluate_baseline_clean_corrupted(
        model=model,
        dataloader=train_dataloader,
        metrics=metrics,
    )

    print(f"Scoring edges with {args.ig_style}")
    attribute(
        model,
        graph,
        train_dataloader,
        partial(task_metric if not args.div_for_scoring else kl_div, reduction="sum", loss=False),
        ig_style=args.ig_style,
        integration_steps=args.integrated_steps,
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

    template_info_df = merge_strategy_dataframes(strategy_dataframes)
    template_info_dfs.append(template_info_df)

if not template_info_dfs:
    raise ValueError("No non-empty template dataloaders were found; nothing to evaluate.")

all_info_df = pd.concat(template_info_dfs, ignore_index=True)

if len(all_info_df) != len(per_sample_eval_df):
    raise ValueError(
        f"Template-averaged dataframe height {len(all_info_df)} does not match "
        f"single-sample dataframe height {len(per_sample_eval_df)}."
    )

all_info_df.to_pickle(all_averaging_score_path)
print(f"Saved all-averaging evaluation dataframe to {all_averaging_score_path}")

elapsed = (time.time() - script_start_time)/60
print(f"Finished in {elapsed:.2f} minutes.")
