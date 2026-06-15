#!/usr/bin/env python3
"""Generate samplewise-variance figures from stored experiment outputs.

This script reads the compact outputs produced by ``pareto_single_sample_analysis.py``
and generates:

1. A two-panel figure with an absolute-score-rank UMAP and a sample-level
   pairwise Jaccard matrix.
2. A seven-panel diagnostic figure for the SVA ``plural_1`` template.

The default configuration targets the public example run:

    results_rephrasing_samplewise_variance/gpt2/sva/graph_eval_with_prob_diff

Only ``individual_score_eval.pkl`` and ``edge_score_array.npz`` are required.
The script does not load a model or recompute attribution scores.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from umap import UMAP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_rephrasing_samplewise_variance"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visualization" / "outputs" / "rephrasing_samplewise_variance"
POINT_BLUE = "#2F5F8F"
TREND_ORANGE = "#D55E00"


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


def natural_key(text: object) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text)))


def format_model_name(model: str) -> str:
    return {"gpt2": "GPT-2 small", "pythia-160m": "Pythia-160M"}.get(model, model)


def format_task_name(task: str) -> str:
    return {"sva": "SVA", "ioi": "IOI", "greater-than": "greater-than"}.get(task, task)


def resolve_result_dir(args: argparse.Namespace) -> Path:
    result_dir = (
        args.results_root
        / args.model
        / args.task
        / args.graph_eval
        / args.circuit
        / args.scoring
        / f"{args.seed}_seed"
    )
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")
    return result_dir


def load_ordered_train_df(args: argparse.Namespace) -> pd.DataFrame:
    """Reconstruct the same row order used by pareto_single_sample_analysis.py."""
    csv_path = PROJECT_ROOT / "data" / args.task / f"{args.model_family}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")

    full_df = pd.read_csv(csv_path)
    if args.max_n_data is not None:
        sample_n = min(args.max_n_data, len(full_df))
        full_df = full_df.sample(n=sample_n, random_state=args.seed, replace=False).reset_index(drop=True)

    if not (0 < args.scoring_sample_number <= len(full_df)):
        raise ValueError(
            f"--scoring-sample-number must be in (0, {len(full_df)}], got {args.scoring_sample_number}"
        )

    raw_train_df = full_df.iloc[: args.scoring_sample_number].reset_index(drop=True)
    template_labels = sorted(full_df["template_label"].unique().tolist())
    grouped = [
        raw_train_df[raw_train_df["template_label"] == template_label].reset_index(drop=True)
        for template_label in template_labels
    ]
    return pd.concat(grouped, ignore_index=True)


def load_samplewise_data(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    ensure_numpy_core_compatibility()
    result_dir = resolve_result_dir(args)
    score_df = pd.read_pickle(result_dir / "individual_score_eval.pkl").reset_index(drop=True)
    with np.load(result_dir / "edge_score_array.npz") as npz_data:
        edge_scores = np.asarray(npz_data["edge_scores"], dtype=np.float32)

    train_df = load_ordered_train_df(args)
    if len(score_df) != len(train_df):
        raise ValueError(f"Result/data length mismatch: {len(score_df)} vs {len(train_df)}")
    if edge_scores.shape[0] != len(score_df):
        raise ValueError(f"Score matrix row mismatch: {edge_scores.shape[0]} vs {len(score_df)}")
    return score_df, edge_scores, train_df


def nearest_target_index(score_df: pd.DataFrame, strategy: str, requested_edges: int) -> tuple[int, int]:
    targets = np.asarray(score_df[f"{strategy}_n_edges_target"].iloc[0], dtype=int)
    target_idx = int(np.argmin(np.abs(targets - requested_edges)))
    return target_idx, int(targets[target_idx])


def actual_edge_count(score_df: pd.DataFrame, strategy: str, target_idx: int) -> int:
    real_edges = np.asarray(score_df[f"{strategy}_n_edges_real"].iloc[0], dtype=int)
    return int(real_edges[target_idx])


def abs_rank_matrix(edge_scores: np.ndarray) -> np.ndarray:
    """Return normalized ranks of absolute edge scores, lower rank = larger score."""
    abs_scores = np.abs(edge_scores)
    order = np.argsort(-abs_scores, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    row_ids = np.arange(edge_scores.shape[0])[:, None]
    ranks[row_ids, order] = np.arange(edge_scores.shape[1], dtype=np.float32)
    ranks /= max(1, edge_scores.shape[1] - 1)
    return ranks


def compute_umap_embedding(edge_scores: np.ndarray, *, random_state: int) -> np.ndarray:
    rank_matrix = abs_rank_matrix(edge_scores)
    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(rank_matrix), dtype=np.float32)


def template_color_and_marker(labels: Iterable[str]) -> tuple[dict[str, object], dict[str, str]]:
    labels = sorted(set(map(str, labels)), key=natural_key)
    base_ids = sorted({label.split("_", 1)[1] if "_" in label else label for label in labels}, key=natural_key)
    color_sequence = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        color_sequence.extend(cmap(i) for i in range(cmap.N))
    base_colors = {base: color_sequence[i % len(color_sequence)] for i, base in enumerate(base_ids)}
    colors = {label: base_colors[label.split("_", 1)[1] if "_" in label else label] for label in labels}
    markers = {label: ("^" if label.startswith("plural_") else "o") for label in labels}
    return colors, markers


def template_sort_key(label: str) -> tuple[object, ...]:
    match = re.fullmatch(r"(single|plural)_(\d+)", label)
    if match is None:
        return (1, *natural_key(label))
    prefix, number = match.groups()
    priority_nums = {2: 0, 3: 1, 4: 2, 6: 3, 7: 4}
    prefix_rank = 0 if prefix == "single" else 1
    number = int(number)
    if number in priority_nums:
        return (0, priority_nums[number], prefix_rank)
    return (1, *natural_key(label))


def reorder_by_template(train_df: pd.DataFrame) -> np.ndarray:
    labels = train_df["template_label"].astype(str).tolist()
    return np.array(sorted(range(len(labels)), key=lambda idx: (template_sort_key(labels[idx]), idx)), dtype=int)


def selected_edge_sets(score_df: pd.DataFrame, *, strategy: str, target_idx: int) -> list[np.ndarray]:
    selected_col = f"{strategy}_selected_edges_id"
    return [np.asarray(steps[target_idx], dtype=np.int64) for steps in score_df[selected_col]]


def pairwise_jaccard(edge_sets: list[np.ndarray]) -> np.ndarray:
    unique_sets = [np.unique(edge_ids) for edge_ids in edge_sets]
    lengths = np.asarray([len(edge_ids) for edge_ids in unique_sets], dtype=np.float32)
    n_samples = len(unique_sets)
    max_edge = max((int(edge_ids.max()) for edge_ids in unique_sets if len(edge_ids) > 0), default=-1)
    if max_edge < 0:
        return np.ones((n_samples, n_samples), dtype=np.float32)

    rows = []
    cols = []
    for row_idx, edge_ids in enumerate(unique_sets):
        if len(edge_ids) == 0:
            continue
        rows.append(np.full(len(edge_ids), row_idx, dtype=np.int32))
        cols.append(edge_ids.astype(np.int64, copy=False))

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    indicator = sparse.csr_matrix(
        (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
        shape=(n_samples, max_edge + 1),
    )
    intersections = (indicator @ indicator.T).toarray().astype(np.float32)
    unions = lengths[:, None] + lengths[None, :] - intersections
    return np.divide(intersections, unions, out=np.ones_like(intersections), where=unions > 0)


def template_boundaries(labels: pd.Series) -> tuple[list[int], list[float], list[str]]:
    values = labels.astype(str).reset_index(drop=True)
    boundaries = [0]
    centers: list[float] = []
    names: list[str] = []
    start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values.iloc[idx] != values.iloc[start]:
            boundaries.append(idx)
            centers.append((start + idx - 1) / 2)
            names.append(values.iloc[start])
            start = idx
    return boundaries, centers, names


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=n/a"
    if p_value < 1e-3:
        return f"p={p_value:.1e}"
    return f"p={p_value:.3f}"


def spearman_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    result = spearmanr(x[mask], y[mask])
    statistic = getattr(result, "statistic", result.correlation)
    return float(statistic), float(result.pvalue)


def rolling_median_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 5:
        return np.array([]), np.array([])
    order = np.argsort(x_fit)
    xs = x_fit[order]
    ys = y_fit[order]
    window = max(5, min(17, (len(xs) // 5) * 2 + 1))
    half = window // 2
    smooth_x = []
    smooth_y = []
    for idx in range(len(xs)):
        lo = max(0, idx - half)
        hi = min(len(xs), idx + half + 1)
        smooth_x.append(float(np.median(xs[lo:hi])))
        smooth_y.append(float(np.median(ys[lo:hi])))
    return np.asarray(smooth_x), np.asarray(smooth_y)


def pessimistic_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(matrix[:, ::-1], axis=1)[:, ::-1]


def first_momentum(edge_scores: np.ndarray) -> np.ndarray:
    weights = np.arange(1, edge_scores.shape[1] + 1, dtype=np.float64)
    out = np.empty(edge_scores.shape[0], dtype=np.float64)
    for idx, row in enumerate(edge_scores):
        abs_scores = np.sort(np.abs(row).astype(np.float64))[::-1]
        total = float(abs_scores.sum())
        out[idx] = np.nan if total <= 0 or not np.isfinite(total) else float(abs_scores @ weights / total)
    return out


def included_mass(edge_scores: np.ndarray, edge_sets: list[np.ndarray]) -> np.ndarray:
    out = np.empty(edge_scores.shape[0], dtype=np.float64)
    for idx, edge_ids in enumerate(edge_sets):
        abs_scores = np.abs(edge_scores[idx].astype(np.float64))
        total = float(abs_scores.sum())
        if total <= 0 or not np.isfinite(total):
            out[idx] = np.nan
        elif len(edge_ids) == 0:
            out[idx] = 0.0
        else:
            out[idx] = float(abs_scores[np.asarray(edge_ids, dtype=int)].sum() / total)
    return out


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def draw_umap(ax: plt.Axes, embedding: np.ndarray, train_df: pd.DataFrame) -> None:
    labels = sorted(train_df["template_label"].astype(str).unique(), key=natural_key)
    colors, markers = template_color_and_marker(labels)
    for label in labels:
        mask = train_df["template_label"].astype(str).to_numpy() == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.82,
            marker=markers[label],
            color=colors[label],
            edgecolors="none",
            label=label,
        )
    ax.set_xlabel("Absolute-score rank UMAP 1")
    ax.set_ylabel("Absolute-score rank UMAP 2")
    ax.grid(True, linestyle=":", color="#D2D2D2", linewidth=0.55)

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=legend_labels,
        title="template_label",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=5.8,
        title_fontsize=7.0,
        frameon=False,
        borderaxespad=0.0,
        ncol=1,
    )


def draw_jaccard_matrix(
    ax: plt.Axes,
    jaccard: np.ndarray,
    train_df: pd.DataFrame,
    *,
    target_edges: int,
    real_edges: int,
) -> None:
    order = reorder_by_template(train_df)
    matrix = jaccard[np.ix_(order, order)].copy()
    ordered_df = train_df.iloc[order].reset_index(drop=True)

    lower_mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    shown = np.ma.array(matrix, mask=lower_mask)
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    vmin = float(np.nanmin(off_diag))
    vmax = float(np.nanmax(off_diag))
    im = ax.imshow(shown, cmap="viridis", norm=Normalize(vmin=vmin, vmax=vmax), interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Pairwise Jaccard index")

    boundaries, centers, names = template_boundaries(ordered_df["template_label"])
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.5)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.5)
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(names, rotation=90, fontsize=4.9)
    ax.set_yticklabels(names, fontsize=4.9)
    ax.set_xlabel("Template")
    ax.set_ylabel("Template")
    ax.text(
        0.98,
        0.04,
        f"target edges = {target_edges}\nreal edges = {real_edges}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.92},
    )


def plot_umap_and_pji(
    score_df: pd.DataFrame,
    edge_scores: np.ndarray,
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    target_idx: int,
    target_edges: int,
    real_edges: int,
) -> list[Path]:
    configure_style()
    embedding = compute_umap_embedding(edge_scores, random_state=args.umap_seed)
    edge_sets = selected_edge_sets(score_df, strategy=args.strategy, target_idx=target_idx)
    jaccard = pairwise_jaccard(edge_sets)

    fig, axes = plt.subplots(1, 2, figsize=(13.1, 5.6), gridspec_kw={"width_ratios": [1.08, 1.0]})
    draw_umap(axes[0], embedding, train_df)
    draw_jaccard_matrix(axes[1], jaccard, train_df, target_edges=target_edges, real_edges=real_edges)
    fig.suptitle(f"Template-dependent sample circuits for {format_model_name(args.model)} on {format_task_name(args.task)}")
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0)
    return save_figure(fig, args.output_dir / f"{args.model}_{args.task}_template_circuit_umap_pji", args.formats)


def draw_scatter_with_spearman(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    color: str = POINT_BLUE,
    log_y: bool = False,
) -> None:
    ax.scatter(x, y, s=20, color=color, alpha=0.78, edgecolors="white", linewidths=0.3)
    xs, ys = rolling_median_curve(x, y)
    if len(xs):
        ax.plot(xs, ys, color=TREND_ORANGE, linewidth=1.45, label="rolling median")
    rho, p_value = spearman_stats(x, y)
    ax.text(
        0.97,
        0.95,
        rf"$\rho={rho:.2f}$" + "\n" + format_p_value(p_value),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CFCFCF", "alpha": 0.94},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        positive = y[np.isfinite(y) & (y > 0)]
        if positive.size:
            ax.set_yscale("log")
    ax.grid(True, linestyle=":", color="#D2D2D2", linewidth=0.55)


def diagnostic_arrays(
    score_df: pd.DataFrame,
    edge_scores: np.ndarray,
    *,
    strategy: str,
    target_idx: int,
) -> dict[str, np.ndarray]:
    faithfulness = np.stack(score_df[f"{strategy}_metric_faithfulness_train"].map(np.asarray).to_numpy()).astype(float)
    raw_q = score_df["metric_clean"].to_numpy(dtype=float) - score_df["metric_corrupted"].to_numpy(dtype=float)
    abs_q = np.abs(raw_q)
    u = np.abs(faithfulness - 1.0)
    ubar = pessimistic_matrix(u)
    u_prime = np.abs((faithfulness - 1.0) * raw_q[:, None])
    ubar_prime = pessimistic_matrix(u_prime)
    edge_sets = selected_edge_sets(score_df, strategy=strategy, target_idx=target_idx)
    return {
        "abs_q": abs_q,
        "u": u[:, target_idx],
        "ubar": ubar[:, target_idx],
        "u_prime": u_prime[:, target_idx],
        "ubar_prime": ubar_prime[:, target_idx],
        "mu": first_momentum(edge_scores),
        "r": included_mass(edge_scores, edge_sets),
    }


def plot_distribution_panel(ax: plt.Axes, values: np.ndarray) -> None:
    values = values[np.isfinite(values) & (values > 0)]
    rng = np.random.default_rng(0)
    jitter = np.clip(rng.normal(loc=0.0, scale=0.026, size=len(values)), -0.075, 0.075)
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    top_point_indices = np.argsort(values)[-3:][::-1]

    ax.scatter(jitter, values, s=20, color=POINT_BLUE, alpha=0.76, edgecolors="white", linewidths=0.35, zorder=3)
    ax.hlines(median, -0.115, 0.115, color="#111111", linewidth=1.15, zorder=4)
    ax.vlines(0.14, q25, q75, color="#111111", linewidth=1.15, zorder=4)
    ax.hlines([q25, q75], 0.115, 0.165, color="#111111", linewidth=1.15, zorder=4)

    for idx in top_point_indices:
        ax.annotate(
            f"{values[idx]:.3g}",
            xy=(jitter[idx], values[idx]),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.6,
            color="#1F3A55",
        )
    for y_value, text, x_value in (
        (median, f"median {median:.3g}", 0.115),
        (q75, f"Q3 {q75:.3g}", 0.165),
        (q25, f"Q1 {q25:.3g}", 0.165),
    ):
        ax.annotate(
            text,
            xy=(x_value, y_value),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.6,
            color="#111111",
        )

    ax.set_xlim(-0.22, 0.34)
    ax.set_xticks([])
    ax.set_ylabel("U")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", color="#D2D2D2", linewidth=0.55)


def plot_seven_panel_diagnostics(
    score_df: pd.DataFrame,
    edge_scores: np.ndarray,
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    target_idx: int,
    target_edges: int,
) -> list[Path]:
    arrays = diagnostic_arrays(score_df, edge_scores, strategy=args.strategy, target_idx=target_idx)
    template_mask = train_df["template_label"].astype(str).to_numpy() == args.template_label
    if not np.any(template_mask):
        raise ValueError(f"No rows found for template {args.template_label!r}")

    configure_style()
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8))
    flat = axes.ravel()
    for ax in flat:
        ax.set_axisbelow(True)

    plot_distribution_panel(flat[0], arrays["u"][template_mask])
    flat[0].set_title(f"{args.template_label}: U distribution")
    flat[0].text(
        0.95,
        0.95,
        f"target edges = {target_edges}",
        transform=flat[0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CFCFCF", "alpha": 0.94},
    )

    draw_scatter_with_spearman(
        flat[1],
        arrays["abs_q"][template_mask],
        arrays["ubar"][template_mask],
        xlabel=r"$|Q_G|$",
        ylabel=r"$\bar{U}$",
        color=POINT_BLUE,
        log_y=True,
    )
    flat[1].set_title(r"$\bar{U}$ vs. $|Q_G|$")

    draw_scatter_with_spearman(
        flat[2],
        arrays["abs_q"][template_mask],
        arrays["ubar_prime"][template_mask],
        xlabel=r"$|Q_G|$",
        ylabel=r"$\bar{U}'$",
        color=POINT_BLUE,
        log_y=False,
    )
    flat[2].set_title(r"$\bar{U}'$ vs. $|Q_G|$")

    draw_scatter_with_spearman(
        flat[3],
        arrays["abs_q"][template_mask],
        arrays["mu"][template_mask],
        xlabel=r"$|Q_G|$",
        ylabel=r"$\mu$",
        color=POINT_BLUE,
    )
    flat[3].set_title(r"$\mu$ vs. $|Q_G|$")

    draw_scatter_with_spearman(
        flat[4],
        arrays["mu"][template_mask],
        arrays["r"][template_mask],
        xlabel=r"$\mu$",
        ylabel=r"$R$",
        color=POINT_BLUE,
    )
    flat[4].set_title(r"$R$ vs. $\mu$")

    draw_scatter_with_spearman(
        flat[5],
        arrays["r"][template_mask],
        arrays["ubar"][template_mask],
        xlabel=r"$R$",
        ylabel=r"$\bar{U}$",
        color=POINT_BLUE,
        log_y=True,
    )
    flat[5].set_title(r"$\bar{U}$ vs. $R$")

    draw_scatter_with_spearman(
        flat[6],
        arrays["mu"][template_mask],
        arrays["ubar"][template_mask],
        xlabel=r"$\mu$",
        ylabel=r"$\bar{U}$",
        color=POINT_BLUE,
        log_y=True,
    )
    flat[6].set_title(r"$\bar{U}$ vs. $\mu$")
    flat[7].axis("off")

    fig.suptitle(
        f"{format_model_name(args.model)} on {format_task_name(args.task)} template {args.template_label}",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=1.0, h_pad=1.0)

    summary = pd.DataFrame(
        {
            "graph_id": np.flatnonzero(template_mask),
            "abs_q": arrays["abs_q"][template_mask],
            "u": arrays["u"][template_mask],
            "ubar": arrays["ubar"][template_mask],
            "u_prime": arrays["u_prime"][template_mask],
            "ubar_prime": arrays["ubar_prime"][template_mask],
            "mu": arrays["mu"][template_mask],
            "r": arrays["r"][template_mask],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / f"{args.model}_{args.task}_{args.template_label}_diagnostics_{target_edges}_edges.csv", index=False)
    return save_figure(
        fig,
        args.output_dir / f"{args.model}_{args.task}_{args.template_label}_seven_panel_diagnostics",
        args.formats,
    )


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> list[Path]:
    written = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        kwargs = {"dpi": 350} if ext in {"png", "jpg", "jpeg"} else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-family", default="gpt2", help="Dataset CSV family name.")
    parser.add_argument("--task", default="sva")
    parser.add_argument("--graph-eval", default="graph_eval_with_prob_diff")
    parser.add_argument("--circuit", default="ceap_20")
    parser.add_argument("--scoring", default="metric_scored")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-n-data", type=int, default=2000)
    parser.add_argument("--scoring-sample-number", type=int, default=1000)
    parser.add_argument("--strategy", default="greedy", choices=("greedy", "topn"))
    parser.add_argument("--target-edges", type=int, default=1000)
    parser.add_argument("--template-label", default="plural_1")
    parser.add_argument("--umap-seed", type=int, default=0)
    parser.add_argument("--formats", nargs="+", default=["pdf"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_df, edge_scores, train_df = load_samplewise_data(args)
    target_idx, target_edges = nearest_target_index(score_df, args.strategy, args.target_edges)
    real_edges = actual_edge_count(score_df, args.strategy, target_idx)

    written = []
    written.extend(
        plot_umap_and_pji(
            score_df,
            edge_scores,
            train_df,
            args,
            target_idx=target_idx,
            target_edges=target_edges,
            real_edges=real_edges,
        )
    )
    written.extend(
        plot_seven_panel_diagnostics(
            score_df,
            edge_scores,
            train_df,
            args,
            target_idx=target_idx,
            target_edges=target_edges,
        )
    )

    print(f"Selected target edge count: {target_edges} (real edge count: {real_edges})")
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
