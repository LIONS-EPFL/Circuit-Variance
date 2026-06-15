import os
import subprocess
import sys
from argparse import  Namespace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule
from ceap.utils import model2family
from ceap.evaluate_graph import evaluate_baseline
from dataset import EAPDataset, make_train_test_dataframe
import warnings
from metrics import get_metric
from numpy_core_compat import ensure_numpy_core_compatibility

from tqdm import tqdm
from ceap.evaluate_graph import evaluate_graph

# Make sure pandas can unpickle arrays saved with legacy numpy._core names.
ensure_numpy_core_compatibility()

def resolve_task_csv_path(task: str, model_name: str, dir_prependix: str = "") -> str:
    family = model2family(model_name)
    candidate_paths = [
        f"{dir_prependix}data/{task}/{family}.csv",
        f"{dir_prependix}data_sparsity/{task}/{family}.csv",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find dataset CSV for task={task!r}, model={model_name!r}. "
        f"Tried: {candidate_paths}"
    )

def normalize_ig_style(args: Namespace) -> None:
    """Normalize `ig_style` and `integrated_steps` in place."""
    args.ig_style = args.ig_style.lower()
    if args.integrated_steps == 1 or args.ig_style == "eap":
        args.ig_style = "eap"
        args.integrated_steps = 1
        warnings.warn("ig_style set to be eap, integrated steps set to be 1.")
    elif args.ig_style in ["ceap","eap-ig"]:
        pass
    else:
        raise ValueError("ig_style not supported! It should be eap, ceap, or eap-ig.")

N_EDGE_TARGET_STEPS = 5
def _edge_target_bounds_for_small_model(task: str) -> Tuple[int, int]:
    """Helper function for choosing edge-count bounds for small models."""
    bounds = {
        "ioi": (10, 6000),
        "gender-bias": (10, 2000),
        "greater-than": (10, 1000),
        "fact-retrieval-comma": (10, 1000),
        "sva": (10, 2000),
        "hypernymy-comma": (10, 5000),
    }
    if task not in bounds:
        raise KeyError(f"Unknown task '{task}' for n_edges target schedule.")
    return bounds[task]


def _edge_target_bounds_for_model(model: str, task: str, edge_number: Optional[int] = None) -> Tuple[int, int]:
    """Helper function for choosing edge-count bounds for a model/task pair."""
    if model in ("gpt2", "EleutherAI/pythia-160m"):
        return _edge_target_bounds_for_small_model(task)
    if model == "gpt2-medium":
        bounds = {
            "sva": (50, 6000),
            "greater-than": (50, 3000),
            "ioi": (50, 30000),
        }
    elif model == "gpt2-large":
        bounds = {
            "sva": (50, 15000),
            "greater-than": (50, 12000),
            "ioi": (50, 50000),
        }
    elif model == "gpt2-xl":
        bounds = {
            "sva": (800, 30000),
            "greater-than": (800, 20000),
            "ioi": (1000, 160000),
        }
    elif model == "EleutherAI/pythia-2.8b":
        bounds = {
            "sva": (800, 30000),
            "greater-than": (800, 20000),
            "ioi": (1000, 80000),
        }
    elif model.startswith(("csp_", "dense1_")) or "circuit-sparsity" in model:
        if edge_number is None:
            raise ValueError("edge_number must be provided for circuit-sparsity models; use len(graph.edges).")
        stop = int(edge_number / 10)
        start = min(max(1, int(edge_number / 100)), stop)
        return start, stop
    else:
        raise NotImplementedError

    if task not in bounds:
        raise NotImplementedError
    return bounds[task]

def get_n_edges_target_list(
    model: str,
    task: str,
    edge_number: Optional[int] = None,
    steps: int = N_EDGE_TARGET_STEPS,
) -> Sequence[int]:
    start, stop = _edge_target_bounds_for_model(model, task, edge_number=edge_number)
    return np.linspace(start, stop, steps, dtype=int).tolist()

def build_result_dir(args: Namespace,) -> str:
    normalize_ig_style(args) # need this to normalise the args fields, just a fail-safe thing.
    metric_kl_str = "kl" if args.div_for_scoring else "metric"
    model_label = str(args.model).replace("\\", "/").rstrip("/").split("/")[-1] # for pythia model: EleutherAI/pythia-160m
    result_dir = os.path.join(
        args.result_folder,
        model_label,
        args.task,
        f"graph_eval_with_{args.metric}",
        f"{args.ig_style}_{args.integrated_steps}",
        f"{metric_kl_str}_scored",
        f"{args.seed}_seed"
    )
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def run_parallel_scoring_subprocesses(
    args: Namespace,
    script_path: Path,
    result_dir: str,
    score_output_template: str = "edge_scores_temp_{idx}.npz",
    dataframe_output_template: str = "{idx}_eval_df.pkl"
) -> Tuple[List[str], List[str]]:
    """Launch per-sample scoring subprocesses and return score/dataframe output paths."""
    score_output_paths = [
        os.path.join(result_dir, score_output_template.format(idx=idx))
        for idx in range(args.scoring_sample_number)
    ]
    dataframe_output_paths = [
        os.path.join(result_dir, dataframe_output_template.format(idx=idx))
        for idx in range(args.scoring_sample_number)
    ]
    if not getattr(args, "continue_run", False):
        for path in score_output_paths+dataframe_output_paths:
            if os.path.exists(path):
                os.remove(path)

    base_cmd = [
        sys.executable,
        str(script_path.resolve()),
        "--model",
        args.model,
        "--task",
        args.task,
        "--metric",
        args.metric,
        "--evaluating-batch-size",
        str(args.evaluating_batch_size),
        "--scoring-sample-number",
        str(args.scoring_sample_number),
        "--ig-style",
        args.ig_style,
        "--integrated-steps",
        str(args.integrated_steps),
        "--max-n-data",
        str(args.max_n_data),
        "--seed",
        str(args.seed),
        "--result-folder",
        args.result_folder,
    ]
    if getattr(args, "div_for_scoring", False):
        base_cmd.append("--div-for-scoring")
    if hasattr(args, "circuit_selection_strategy"):
        base_cmd.extend(
            [
                "--circuit-selection-strategy",
                args.circuit_selection_strategy,
            ]
        )
    if getattr(args, "continue_run", False):
        base_cmd.append("--continue")

    total_job_list = list(range(args.scoring_sample_number))
    job_list_list = [
        total_job_list[i : i + args.parallel_batch_size]
        for i in range(0, len(total_job_list), args.parallel_batch_size)
    ]
    pbar = tqdm(total=len(total_job_list), desc="scoring process")
    for job_list in job_list_list:
        processes = []
        for operation_id in job_list:
            cmd = base_cmd + ["--operation-id", str(operation_id)]
            processes.append(
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

        for proc in processes:
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"Subprocess {proc.args} failed with code {ret}")
        pbar.update(len(job_list))
    pbar.close()
    return score_output_paths, dataframe_output_paths

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
        return "mps"

    return "cpu"


def resolve_model_device(device: Optional[str] = None) -> str:
    """
    Resolve a valid device string for HookedTransformer.from_pretrained.
    Supports explicit device override and automatic fallback.
    """
    if device is None:
        return _auto_device()

    requested = str(device).lower()
    if requested == "auto":
        return _auto_device()

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        warnings.warn("Requested device='cuda' but CUDA is unavailable; falling back to auto device selection.")
        return _auto_device()

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
            return "mps"
        warnings.warn("Requested device='mps' but MPS is unavailable; falling back to cpu.")
        return "cpu"

    if requested == "cpu":
        return "cpu"

    raise ValueError("Unsupported device. Use one of: 'auto', 'cuda', 'mps', 'cpu'.")


def load_model(model_name: str, device: Optional[str] = None) -> HookedRootModule:
    resolved_device = resolve_model_device(device)
    from circuit_sparse_adapter import (
        is_circuit_sparsity_model_identifier,
        load_circuit_sparse_model,
    )

    if is_circuit_sparsity_model_identifier(model_name):
        model = load_circuit_sparse_model(model_name, device=resolved_device)
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.use_attn_in = True
        return model

    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=resolved_device,
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


def prepare_dataloaders_for_different_templates(
    task: str,
    model_name: str,
    max_n_data: int,
    scoring_sample_number: int,
    evaluating_batch_size: int,
    seed: int,
    return_df = False,
    return_sample_dataloader = False,
    do_test = False,
    dir_prependix = ''):
    """Split data by template label."""
    csv_name = resolve_task_csv_path(task, model_name, dir_prependix=dir_prependix)
    full_df = pd.read_csv(csv_name)
    if max_n_data is not None:
        sample_n = min(max_n_data, len(full_df))
        full_df = full_df.sample(n=sample_n, random_state=seed, replace=False).reset_index(drop=True)
    template_label_ls = sorted(full_df.template_label.unique().tolist())
    raw_train_df, raw_test_df = make_train_test_dataframe(
        full_df,
        train_ratio=None,
        train_sample_number=scoring_sample_number,
    ) 

    # Build one train dataset/dataloader per template_label in sorted template_label_ls order.
    template_train_df_ls = [
        raw_train_df[raw_train_df.template_label == template_label].reset_index(drop=True)
        for template_label in template_label_ls
    ]
    train_ds_ls = [EAPDataset(task, template_train_df) for template_train_df in template_train_df_ls]
    empty_train_labels = [
        template_label for template_label, train_ds in zip(template_label_ls, train_ds_ls) if len(train_ds) == 0
    ]
    if empty_train_labels:
        warnings.warn(
            f"Empty template groups in train split: {empty_train_labels}. "
            "Use a bigger scoring_sample_number."
        )
    template_train_dataloader_ls = [
        train_ds.to_dataloader(evaluating_batch_size) for train_ds in train_ds_ls
    ]

    template_test_df_ls = [
        raw_test_df[raw_test_df.template_label == template_label].reset_index(drop=True)
        for template_label in template_label_ls
    ]
    test_ds_ls = [EAPDataset(task, template_test_df) for template_test_df in template_test_df_ls]
    # Most of the time we don't do testing. Test do not give more insights.
    if do_test:
        empty_test_labels = [
            template_label for template_label, test_ds in zip(template_label_ls, test_ds_ls) if len(test_ds) == 0
        ]
        if empty_test_labels:
            warnings.warn(
                f"Empty template groups in test split: {empty_test_labels}. "
                "Adjust max_n_data/scoring_sample_number or use a stratified split by template_label."
            )
    template_test_dataloader_ls = [
        test_ds.to_dataloader(evaluating_batch_size) for test_ds in test_ds_ls
    ]

    if not (return_df or return_sample_dataloader):
        return template_train_dataloader_ls, template_test_dataloader_ls, template_label_ls
    else:
        train_df = pd.concat(template_train_df_ls, ignore_index=True)
        test_df = pd.concat(template_test_df_ls, ignore_index=True)
    if return_sample_dataloader:
        sample_train_dataloader_ls = [
            EAPDataset(task, train_df.iloc[i:i+1]).to_dataloader(1)
            for i in range(len(train_df))]
        sample_test_dataloader_ls = [
            EAPDataset(task, test_df.iloc[i:i+1]).to_dataloader(1)
            for i in range(len(test_df))]
    if return_df and not return_sample_dataloader:
        return (
                train_df,
                test_df,
                template_train_dataloader_ls,
                template_test_dataloader_ls,
                template_label_ls,
                )
    if not return_df and return_sample_dataloader:
        return (
            sample_train_dataloader_ls,
            sample_test_dataloader_ls,
            template_train_dataloader_ls,
            template_test_dataloader_ls,
            template_label_ls,
                )
    if return_df and return_sample_dataloader:
        return (
                train_df,
                test_df,
                sample_train_dataloader_ls,
                sample_test_dataloader_ls,
                template_train_dataloader_ls,
                template_test_dataloader_ls,
                template_label_ls,
                )


def get_task_metrics(task_metric_name: str, task: str, model: HookedRootModule):
    task_metric = get_metric(task_metric_name, task, model=model)
    kl_div = get_metric("kl_divergence", task, model=model)
    return task_metric, kl_div


def evaluate_baseline_clean_corrupted(model: HookedRootModule, dataloader, metrics):
    baselines = evaluate_baseline(model, dataloader, metrics)
    corrupted_baselines = evaluate_baseline(model, dataloader, metrics, run_corrupted=True)
    return baselines, corrupted_baselines


def compute_faithfulness(ablated_performance, baseline, corrupted_baseline, epsilon: float=1e-8):
    def _safe_denominator(denominator):
        sign = torch.sign(denominator)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return sign * torch.clamp(denominator.abs(), min=epsilon)
    faithfulness = (ablated_performance - corrupted_baseline) / _safe_denominator(baseline - corrupted_baseline)
    return faithfulness

def circuit_selection_data_collection_per_sample(
    n_edges_target_ls,
    graph,
    model,
    train_dataloader,
    metrics_dict,
    train_baselines,
    train_corrupted_baselines,
    strategy,
    save_graph=False,
    graph_dir=None,
):
    """Evaluate per-sample faithfulness for circuits at multiple edge counts.

    Applies either greedy or top-n circuit selection for each target edge count,
    prunes disconnected nodes, evaluates the selected circuit on the dataloader,
    and returns a dataframe with one row per sample. Step-dependent fields such as
    target edge counts, realized edge counts, selected edge ids, and faithfulness
    values are stored as arrays in each row.
    """
    if strategy not in {"greedy", "topn"}:
        raise ValueError(f"Unsupported strategy '{strategy}'. Expected one of: ['greedy', 'topn'].")

    metrics = list(metrics_dict.values())
    metric_names = list(metrics_dict.keys())
    n_edges_already = 0
    pbar = tqdm(total=n_edges_target_ls[-1], desc="Selected edges")
    metrics_history = []
    n_edges_real = []
    selected_edge_ids_list = []
    for n_edges_target in n_edges_target_ls:
        if strategy == "greedy":
            graph.apply_greedy(
                n_edges_target=n_edges_target,
                n_edges_already=n_edges_already,
                reset=n_edges_already == 0,
            )
            unpruned_state = graph.snapshot_selection_state()
        else:
            graph.apply_topn(n=n_edges_target, absolute=True)
            unpruned_state = None

        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
        n = graph.count_included_edges()
        selected_edge_ids = graph.selected_edge_ids().detach().cpu().numpy()

        metrics_history.append(
            compute_per_sample_metrics_step(
                model=model,
                graph=graph,
                train_dataloader=train_dataloader,
                metrics=metrics,
                train_baselines=train_baselines,
                train_corrupted_baselines=train_corrupted_baselines,
            )
        )
        n_edges_real.append(n)
        selected_edge_ids_list.append(selected_edge_ids)
        if save_graph:
            if graph_dir is None:
                raise ValueError(
                    f"graph_dir must be provided when save_graph=True for {strategy} per-sample collection."
                )
            os.makedirs(graph_dir, exist_ok=True)
            json_path = os.path.join(
                graph_dir,
                f"target{n_edges_target}_real{n}.json",
            )
            graph.to_json(json_path)

        if strategy == "greedy":
            graph.restore_selection_state(unpruned_state)

        pbar.update(n_edges_target - n_edges_already)
        n_edges_already = n_edges_target

    pbar.close()
    return build_per_sample_metrics_dataframe(
        metrics_history=metrics_history,
        n_edges_target_ls=n_edges_target_ls,
        n_edges_real=n_edges_real,
        selected_edge_ids_list=selected_edge_ids_list,
        train_baselines=train_baselines,
        train_corrupted_baselines=train_corrupted_baselines,
        column_prefix=strategy,
        metric_names=metric_names,
    )

def data_collection_per_sample(
    model,
    graph,
    train_dataloader,
    metrics,
    train_baselines,
    train_corrupted_baselines,
):
    """Evaluate the current graph and return per-sample faithfulness values.

    Runs the ablated graph on the dataloader, compares each metric against its
    clean and corrupted baselines, and returns one faithfulness tensor per metric.
    If `metrics` is a single callable, returns a single tensor instead of a list.
    """
    ablated_performances_train = evaluate_graph(model,graph,train_dataloader, metrics, quiet=True,)
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        ablated_performances_train = [ablated_performances_train]
        metrics_list = False

    faithfulnesses_train = []
    for i,_ in enumerate(metrics):
        faithfulness_train = compute_faithfulness(ablated_performance=ablated_performances_train[i],
                                                baseline = train_baselines[i], 
                                                corrupted_baseline = train_corrupted_baselines[i])
        faithfulnesses_train.append(faithfulness_train)

    if not metrics_list: return faithfulnesses_train[0]
    else: return faithfulnesses_train 


def compute_per_sample_metrics_step(
    model,
    graph,
    train_dataloader,
    metrics,
    train_baselines,
    train_corrupted_baselines,
):
    """Return per-metric faithfulness arrays for one circuit-selection step.

    Wraps `data_collection_per_sample` and converts each faithfulness tensor to a
    NumPy array of shape `(n_samples, 1)`, matching the format expected by
    `build_per_sample_metrics_dataframe`.
    """
    faithfulnesses_train = data_collection_per_sample(
        model=model,
        graph=graph,
        train_dataloader=train_dataloader,
        metrics=metrics,
        train_baselines=train_baselines,
        train_corrupted_baselines=train_corrupted_baselines,
    )

    if not isinstance(metrics, list):
        metrics = [metrics]
        faithfulnesses_train = [faithfulnesses_train]

    metrics_step = []
    for i, _ in enumerate(metrics):
        metric_step = faithfulnesses_train[i].detach().cpu().numpy()[:, None]
        metrics_step.append(metric_step)
    return metrics_step


def build_per_sample_metrics_dataframe(
    metrics_history,
    n_edges_target_ls,
    n_edges_real,
    selected_edge_ids_list,
    train_baselines,
    train_corrupted_baselines,
    column_prefix,
    metric_names
):
    """Build the per-sample dataframe from circuit-selection metric history.

    Each row corresponds to one sample. Edge-count fields, selected edge ids, and
    faithfulness values are stored as arrays over circuit-selection steps, while
    clean and corrupted baseline values are stored as scalars.
    """
    assert isinstance(train_baselines, list) and isinstance(train_corrupted_baselines, list)
    assert len(train_baselines) == len(metric_names) == len(train_corrupted_baselines)

    # metrics_history: list[step], each step is list[metric] of arrays shaped (samples, 1)
    step_arrays = []
    for step_metrics in metrics_history:
        step_concat = np.concatenate(step_metrics, axis=1)  # (samples, n_metrics)
        step_arrays.append(step_concat)
    metrics_array = np.stack(step_arrays, axis=0)  # (steps, samples, n_metrics)

    n_edges_target_arr = np.asarray(n_edges_target_ls, dtype=int)
    n_edges_real_arr = np.asarray(n_edges_real, dtype=int)

    def to_numpy_1d(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    train_baseline_arrs = [to_numpy_1d(x) for x in train_baselines]
    train_corrupted_baseline_arrs = [to_numpy_1d(x) for x in train_corrupted_baselines]

    base_names = [
        "faithfulness_train",
    ]
    column_names = []
    for metric_name in metric_names:
        for base_name in base_names:
            column_names.append(f"{column_prefix}_{metric_name}_{base_name}")

    n_edges_target_key = f"{column_prefix}_n_edges_target"
    n_edges_real_key = f"{column_prefix}_n_edges_real"
    selected_edges_id_key = f"{column_prefix}_selected_edges_id"

    records = []
    num_samples = metrics_array.shape[1]
    for sample_idx in range(num_samples):
        record = { # each sample has a record dictionary.
            n_edges_target_key: n_edges_target_arr.copy(),
            n_edges_real_key: n_edges_real_arr.copy(),
            selected_edges_id_key: selected_edge_ids_list,
        }
        for metric_idx, name in enumerate(column_names):
            record[name] = metrics_array[:, sample_idx, metric_idx].copy()
        for i, metric_name in enumerate(metric_names):
            record[f"{metric_name}_clean"] = train_baseline_arrs[i][sample_idx].item()
            record[f"{metric_name}_corrupted"] = train_corrupted_baseline_arrs[i][sample_idx].item()
        records.append(record)

    return pd.DataFrame.from_records(records)


def selected_circuit_strategies(strategy: str) -> tuple[str, ...]:
    if strategy == "both":
        return ("greedy", "topn")
    return (strategy,)


def remove_duplicate_columns(df):
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def merge_strategy_dataframes(strategy_dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected_strategies = list(strategy_dataframes)
    if len(selected_strategies) == 1:
        return strategy_dataframes[selected_strategies[0]].reset_index(drop=True)

    first_strategy = selected_strategies[0]
    first_len = len(strategy_dataframes[first_strategy])
    for strategy in selected_strategies[1:]:
        if len(strategy_dataframes[strategy]) != first_len:
            raise ValueError(
                f"{first_strategy} and {strategy} circuit dataframes must have the same number of rows."
            )

    return remove_duplicate_columns(
        pd.concat(
            [strategy_dataframes[strategy].reset_index(drop=True) for strategy in selected_strategies],
            axis=1,
        )
    )
