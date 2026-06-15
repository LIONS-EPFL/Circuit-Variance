#!/usr/bin/env python3
"""Generate Appendix-style resampling-variance plots.

This script reads the compact outputs produced by ``pareto_variance.py`` and
generates two paper-ready figures:

1. Pairwise Jaccard index between circuits found from different resampling seeds.
2. Mean unfaithfulness as the target number of edges increases.

The default configuration targets the public example run:

    results_resampling_variance/gpt2/sva/graph_eval_with_prob_diff

Only stored ``all_averaging_score_eval.pkl`` files are required; the script does
not load a model or regenerate circuits.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_resampling_variance"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visualization" / "outputs" / "resampling_variance"
RESULT_FILENAME = "all_averaging_score_eval.pkl"
DECODE = "greedy"
SETUP_LABELS = {
    "ceap": "CEAP",
    "eap-ig": "EAP-IG",
    "eap": "EAP",
}
SETUP_COLORS = {
    "CEAP": "#D55E00",
    "EAP-IG": "#0072B2",
    "EAP": "#2B2B2B",
}


@dataclass(frozen=True)
class Curve:
    template: str
    setup: str
    label: str
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray


def ensure_numpy_core_compatibility() -> None:
    """Allow newer NumPy versions to read pickles created under NumPy 2.x."""
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    for name in ("numeric", "multiarray", "umath", "_multiarray_umath"):
        module_name = f"numpy.core.{name}"
        alias_name = f"numpy._core.{name}"
        if alias_name in sys.modules:
            continue
        try:
            __import__(module_name)
        except Exception:
            continue
        sys.modules[alias_name] = sys.modules[module_name]


def natural_key(text: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text)))


def setup_label(setup_name: str) -> str:
    family = setup_name.rsplit("_", 1)[0]
    return SETUP_LABELS.get(family, family.upper())


def format_model_name(model: str) -> str:
    if model == "gpt2":
        return "GPT-2 small"
    return model


def format_task_name(task: str) -> str:
    return {"sva": "SVA", "ioi": "IOI", "greater-than": "greater-than"}.get(task, task)


def format_edge_tick(value: float, _pos: int) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def resolve_eval_dir(results_root: Path, model: str, task: str, eval_dir: str | None) -> Path:
    task_root = results_root / model / task
    if eval_dir is not None:
        path = task_root / eval_dir
        if not path.exists():
            raise FileNotFoundError(f"Evaluation directory does not exist: {path}")
        return path

    candidates = sorted(p for p in task_root.glob("graph_eval_with_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No graph_eval_with_* directory found under {task_root}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"Multiple evaluation directories found under {task_root}: {names}. Use --eval-dir.")
    return candidates[0]


def available_setups(eval_dir: Path, requested: Iterable[str] | None) -> list[str]:
    existing = sorted((p.name for p in eval_dir.iterdir() if p.is_dir()), key=natural_key)
    if requested is None:
        return existing

    requested_list = list(requested)
    missing = [setup for setup in requested_list if setup not in existing]
    if missing:
        raise FileNotFoundError(f"Missing setup(s) under {eval_dir}: {missing}")
    return requested_list


def read_result(path: Path, cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if path not in cache:
        cache[path] = pd.read_pickle(path)
    return cache[path]


def extract_target_to_edges(df: pd.DataFrame, decode: str = DECODE) -> dict[int, set[int]]:
    target_col = f"{decode}_n_edges_target"
    selected_col = f"{decode}_selected_edges_id"
    if target_col not in df or selected_col not in df or df.empty:
        return {}

    targets = np.asarray(df[target_col].iloc[0], dtype=int).reshape(-1)
    selected_steps = df[selected_col].iloc[0]
    n_steps = min(len(targets), len(selected_steps))

    out: dict[int, set[int]] = {}
    for target, edge_ids in zip(targets[:n_steps], selected_steps[:n_steps]):
        edge_arr = np.asarray(edge_ids, dtype=int).reshape(-1)
        out[int(target)] = {int(edge_id) for edge_id in edge_arr.tolist()}
    return out


def pairwise_jaccard_curves(
    eval_dir: Path,
    setups: list[str],
    *,
    score_type: str,
    decode: str,
) -> list[Curve]:
    df_cache: dict[Path, pd.DataFrame] = {}
    curves: list[Curve] = []
    for setup in setups:
        setup_dir = eval_dir / setup / score_type
        if not setup_dir.exists():
            continue
        seed_dirs = sorted(setup_dir.glob("*_seed"), key=lambda p: natural_key(p.name))
        templates = sorted(
            {template_dir.name for seed_dir in seed_dirs for template_dir in seed_dir.iterdir() if template_dir.is_dir()},
            key=natural_key,
        )
        for template in templates:
            seed_edges: list[tuple[str, dict[int, set[int]]]] = []
            for seed_dir in seed_dirs:
                pkl_path = seed_dir / template / RESULT_FILENAME
                if not pkl_path.exists():
                    continue
                target_map = extract_target_to_edges(read_result(pkl_path, df_cache), decode=decode)
                if target_map:
                    seed_edges.append((seed_dir.name, target_map))
            stats = compute_pairwise_jaccard_stats(seed_edges)
            if stats is None:
                continue
            x, mean, std, count = stats
            label = setup_label(setup)
            curves.append(Curve(template, setup, label, x, mean, std, count))
    return curves


def compute_pairwise_jaccard_stats(
    seed_edges: list[tuple[str, dict[int, set[int]]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(seed_edges) < 2:
        return None

    values_by_target: dict[int, list[float]] = defaultdict(list)
    for (_, edges_a), (_, edges_b) in itertools.combinations(seed_edges, 2):
        for target in sorted(set(edges_a).intersection(edges_b)):
            set_a = edges_a[target]
            set_b = edges_b[target]
            union = set_a | set_b
            jaccard = 1.0 if len(union) == 0 else len(set_a & set_b) / len(union)
            values_by_target[target].append(float(jaccard))

    if not values_by_target:
        return None

    x = np.array(sorted(values_by_target), dtype=int)
    mean = np.array([np.mean(values_by_target[target]) for target in x], dtype=float)
    std = np.array([np.std(values_by_target[target], ddof=0) for target in x], dtype=float)
    count = np.array([len(values_by_target[target]) for target in x], dtype=int)
    return x, mean, std, count


def unfaithfulness_curves(
    eval_dir: Path,
    setups: list[str],
    *,
    score_type: str,
    decode: str,
) -> list[Curve]:
    df_cache: dict[Path, pd.DataFrame] = {}
    curves: list[Curve] = []
    faithfulness_col = f"{decode}_metric_faithfulness_train"
    target_col = f"{decode}_n_edges_target"

    for setup in setups:
        setup_dir = eval_dir / setup / score_type
        if not setup_dir.exists():
            continue
        seed_dirs = sorted(setup_dir.glob("*_seed"), key=lambda p: natural_key(p.name))
        templates = sorted(
            {template_dir.name for seed_dir in seed_dirs for template_dir in seed_dir.iterdir() if template_dir.is_dir()},
            key=natural_key,
        )
        for template in templates:
            seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
            for seed_dir in seed_dirs:
                pkl_path = seed_dir / template / RESULT_FILENAME
                if not pkl_path.exists():
                    continue
                df = read_result(pkl_path, df_cache)
                if df.empty or faithfulness_col not in df or target_col not in df:
                    continue
                targets = np.asarray(df[target_col].iloc[0], dtype=int).reshape(-1)
                faithfulness = np.stack(df[faithfulness_col].map(np.asarray).to_numpy()).astype(float)
                n_steps = min(len(targets), faithfulness.shape[1])
                unfaithfulness = np.abs(1.0 - faithfulness[:, :n_steps])
                seed_curves.append((targets[:n_steps], np.nanmean(unfaithfulness, axis=0)))
            stats = aggregate_seed_curves(seed_curves)
            if stats is None:
                continue
            x, mean, std, count = stats
            label = setup_label(setup)
            curves.append(Curve(template, setup, label, x, mean, std, count))
    return curves


def aggregate_seed_curves(
    seed_curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not seed_curves:
        return None

    values_by_target: dict[int, list[float]] = defaultdict(list)
    for x_arr, y_arr in seed_curves:
        for target, value in zip(x_arr, y_arr):
            if np.isfinite(value):
                values_by_target[int(target)].append(float(value))
    if not values_by_target:
        return None

    x = np.array(sorted(values_by_target), dtype=int)
    mean = np.array([np.mean(values_by_target[target]) for target in x], dtype=float)
    std = np.array([np.std(values_by_target[target], ddof=0) for target in x], dtype=float)
    count = np.array([len(values_by_target[target]) for target in x], dtype=int)
    return x, mean, std, count


def group_by_template(curves: list[Curve]) -> dict[str, list[Curve]]:
    grouped: dict[str, list[Curve]] = defaultdict(list)
    for curve in curves:
        grouped[curve.template].append(curve)

    setup_order = {"eap-ig": 0, "ceap": 1, "eap": 2}
    for template, template_curves in grouped.items():
        template_curves.sort(key=lambda c: (setup_order.get(c.setup.rsplit("_", 1)[0], 99), natural_key(c.setup)))
    return dict(sorted(grouped.items(), key=lambda item: natural_key(item[0])))


def shared_ylim(curves: list[Curve], *, lower_bound: float | None, upper_bound: float | None) -> tuple[float, float]:
    values = np.concatenate([curve.mean[np.isfinite(curve.mean)] for curve in curves if np.any(np.isfinite(curve.mean))])
    if values.size == 0:
        return 0.0, 1.0

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    span = max(1e-8, y_max - y_min)
    lower = y_min - 0.12 * span
    upper = y_max + 0.12 * span
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    if upper_bound is not None:
        upper = min(upper_bound, upper)
    if lower >= upper:
        upper = lower + 0.1
    return lower, upper


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def plot_template_grid(
    curves: list[Curve],
    *,
    output_path: Path,
    model: str,
    task: str,
    ylabel: str,
    title_suffix: str,
    max_cols: int,
    ylim: tuple[float, float],
    file_formats: list[str],
    dpi: int,
) -> list[Path]:
    if not curves:
        raise ValueError(f"No curves available for {title_suffix}")

    configure_matplotlib()
    grouped = group_by_template(curves)
    templates = list(grouped)
    ncols = min(max_cols, max(1, len(templates)))
    nrows = math.ceil(len(templates) / ncols)
    fig_width = max(8.2, 2.65 * ncols)
    fig_height = max(3.8, 2.1 * nrows + 0.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False, sharey=True)

    legend_handles: dict[str, object] = {}
    for ax, template in zip(axes.ravel(), templates):
        for curve in grouped[template]:
            color = SETUP_COLORS.get(curve.label, "#555555")
            (line,) = ax.plot(curve.x, curve.mean, color=color, linewidth=1.7, label=curve.label)
            ax.fill_between(curve.x, curve.mean - curve.std, curve.mean + curve.std, color=color, alpha=0.16, linewidth=0)
            legend_handles.setdefault(curve.label, line)
        ax.set_title(template)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(format_edge_tick))
        ax.grid(True, linestyle=":", color="#BDBDBD", linewidth=0.55, alpha=0.85)

    for ax in axes.ravel()[len(templates) :]:
        ax.axis("off")

    for row_idx in range(nrows):
        axes[row_idx, 0].set_ylabel(ylabel)
    for col_idx in range(ncols):
        axes[-1, col_idx].set_xlabel("Number of target edges")

    ordered_labels = [label for label in ("EAP-IG", "CEAP", "EAP") if label in legend_handles]
    fig.legend(
        [legend_handles[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        ncol=len(ordered_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(f"{format_model_name(model)} on {format_task_name(task)}: {title_suffix}", y=0.965, fontsize=12.0)
    fig.tight_layout(rect=(0.02, 0.025, 0.995, 0.93), w_pad=0.75, h_pad=0.9)

    written: list[Path] = []
    for file_format in file_formats:
        out = output_path.with_suffix(f".{file_format}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        written.append(out)
    plt.close(fig)
    return written


def write_summary_csv(curves: list[Curve], path: Path, value_name: str) -> None:
    rows = []
    for curve in curves:
        for target, mean, std, count in zip(curve.x, curve.mean, curve.std, curve.count):
            rows.append(
                {
                    "template": curve.template,
                    "setup": curve.setup,
                    "algorithm": curve.label,
                    "n_edges_target": int(target),
                    f"mean_{value_name}": float(mean),
                    f"std_{value_name}": float(std),
                    "n": int(count),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--task", default="sva")
    parser.add_argument("--eval-dir", default=None, help="Evaluation directory name, e.g. graph_eval_with_prob_diff.")
    parser.add_argument("--score-type", default="metric_scored")
    parser.add_argument("--setups", nargs="*", default=None, help="Setups to plot. Defaults to all available setups.")
    parser.add_argument("--decode", default=DECODE)
    parser.add_argument("--max-cols", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--formats", nargs="+", default=["pdf"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_numpy_core_compatibility()
    eval_dir = resolve_eval_dir(args.results_root, args.model, args.task, args.eval_dir)
    setups = available_setups(eval_dir, args.setups)

    pji_curves = pairwise_jaccard_curves(eval_dir, setups, score_type=args.score_type, decode=args.decode)
    unfaithfulness = unfaithfulness_curves(eval_dir, setups, score_type=args.score_type, decode=args.decode)

    base_name = f"{args.model}_{args.task}"
    output_stem = args.output_dir / base_name

    pji_paths = plot_template_grid(
        pji_curves,
        output_path=output_stem.with_name(f"{base_name}_pairwise_jaccard_vs_n_edges_target"),
        model=args.model,
        task=args.task,
        ylabel="Pairwise Jaccard index",
        title_suffix="pairwise circuit overlap",
        max_cols=args.max_cols,
        ylim=shared_ylim(pji_curves, lower_bound=0.0, upper_bound=1.0),
        file_formats=args.formats,
        dpi=args.dpi,
    )
    unfaithfulness_paths = plot_template_grid(
        unfaithfulness,
        output_path=output_stem.with_name(f"{base_name}_unfaithfulness_vs_n_edges_target"),
        model=args.model,
        task=args.task,
        ylabel="Unfaithfulness",
        title_suffix="unfaithfulness",
        max_cols=args.max_cols,
        ylim=shared_ylim(unfaithfulness, lower_bound=0.0, upper_bound=None),
        file_formats=args.formats,
        dpi=args.dpi,
    )

    write_summary_csv(pji_curves, output_stem.with_name(f"{base_name}_pairwise_jaccard_vs_n_edges_target.csv"), "pairwise_jaccard")
    write_summary_csv(unfaithfulness, output_stem.with_name(f"{base_name}_unfaithfulness_vs_n_edges_target.csv"), "unfaithfulness")

    print("Wrote:")
    for path in [*pji_paths, *unfaithfulness_paths]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
