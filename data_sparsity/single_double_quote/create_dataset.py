import argparse
import re
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from tiktoken import Encoding


DEFAULT_VIZ_PATH = Path(
    "/Users/frank/Research/circuit_sparsity/local_cache/viz/modelsig_d1024_L12/"
    "viz_data_d1024_L12_3560de3421d865b854fc5ff5e333c61272eb441c.pt"
)
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("tinypython_2k.csv")
DEFAULT_SOURCE_URL = (
    "https://openaipublic.blob.core.windows.net/circuit-sparsity/viz/"
    "csp_yolo1/single_double_quote/prune_v2/128/viz_data.pt"
)
DEFAULT_NUM_EXAMPLES = 1000
DEFAULT_VARIANTS_PER_TEMPLATE = 40
TARGET_SNIPPET_TOKENS = 32
TEMPLATE_LABELS = {
    0: "if_clause_1",
    1: "print_stmt_1",
    2: "print_stmt_2",
    3: "print_stmt_3",
    4: "strftime_call",
    5: "constructor_call",
    6: "raise_error",
    7: "function_call_1",
    8: "if_clause_2",
    9: "append_call",
    10: "if_clause_3",
    11: "for_loop_1",
    12: "print_stmt_4",
    13: "method_call_1",
    14: "nested_loop_1",
    15: "string_split",
}
SINGLETON_NUMBERED_TEMPLATE_LABELS = {
    "for_loop_1": "for_loop",
    "function_call_1": "function_call",
    "method_call_1": "method_call",
    "nested_loop_1": "nested_loop",
}
MANUAL_TEMPLATE_OVERRIDES = {
    6: {
        "clean": "raise ValueError('Invalid encoded data",
        "corrupted": 'raise ValueError("Invalid encoded data',
    },
    8: {
        "clean": "if degree not in range(3, n-1):\n    print('Degree should be in the range [3, n-1]",
        "corrupted": 'if degree not in range(3, n-1):\n    print("Degree should be in the range [3, n-1]',
    },
    9: {
        "clean": "if D < 1 or D > 31 or H < 0 or H > 24 or M < 0 or M > 60:\n    results.append('Invalid Input",
        "corrupted": 'if D < 1 or D > 31 or H < 0 or H > 24 or M < 0 or M > 60:\n    results.append("Invalid Input',
    },
    14: {
        "clean": (
            "for vertex in vertex_data:\n"
            "    for point in vertex:\n"
            "        if(point == ()):\n"
            "            print('No point at the current iteration"
        ),
        "corrupted": (
            "for vertex in vertex_data:\n"
            "    for point in vertex:\n"
            "        if(point == ()):\n"
            '            print("No point at the current iteration'
        ),
    },
}


def _template_label_for_quote_style(base_label: str, quote_style: str) -> str:
    label = SINGLETON_NUMBERED_TEMPLATE_LABELS.get(base_label, base_label)
    if quote_style == "single":
        return f"{label}_s"
    if quote_style == "double":
        return f"{label}_d"
    raise ValueError(f"Unexpected quote_style: {quote_style!r}")
TEMPLATE_PLACEHOLDERS = {
    0: {"has_cycle": "[COND]"},
    1: {"Distance from node:": "[PRINT_TEXT]"},
    2: {"Optimized Robot Usage:": "[USAGE_TEXT]"},
    3: {"Output Question 1:": "[QUESTION_TEXT]"},
    4: {"timestamp": "[TIME_VAR]"},
    5: {"project": "[OBJECT_VAR]", "Project A": "[PROJECT_NAME]"},
    6: {"Invalid encoded data": "[ERROR_TEXT]"},
    7: {"clear_grid": "[FUNC_NAME]", "mine_file.txt": "[FILE_NAME]"},
    8: {"n": "[LIMIT_VAR]"},
    9: {"results": "[RESULT_LIST]"},
    10: {"alt": "[ALT_VAR]", "dist": "[DIST_VAR]", "v": "[NODE_VAR]"},
    11: {"solution": "[ITEM_VAR]", "solution_set": "[ITEM_SET]"},
    12: {"No more unsatisfied points": "[STATUS_TEXT]"},
    13: {"graph": "[GRAPH_VAR]", "bigger_data.csv": "[GRAPH_FILE]"},
    14: {"vertex": "[OUTER_ITEM]", "vertex_data": "[OUTER_SET]", "point": "[INNER_ITEM]"},
    15: {
        "convert_to_set": "[FUNC_NAME]",
        "activity": "[ARG_VAR]",
        "start": "[LEFT_VAR]",
        "end": "[RIGHT_VAR]",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deterministic single-vs-double-quote data for sparse transformer experiments."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--viz-path",
        type=Path,
        default=DEFAULT_VIZ_PATH,
        help="Local path to the cached viz_data.pt containing task samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the generated CSV.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help="Final number of rows to export after deterministic sampling.",
    )
    parser.add_argument(
        "--variants-per-template",
        type=int,
        default=DEFAULT_VARIANTS_PER_TEMPLATE,
        help=(
            "How many naming variants to create for each of the 16 base templates "
            "before sampling the final dataset."
        ),
    )
    return parser.parse_args()


def _load_tinypython_encoding() -> Encoding:
    repo_root = Path(__file__).resolve().parents[3] / "circuit_sparsity"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from circuit_sparsity.tiktoken_ext import tinypython

    return Encoding(**tinypython.tinypython_2k())


def _load_viz(viz_path: Path):
    if not viz_path.exists():
        raise FileNotFoundError(
            f"Could not find viz_data file at {viz_path}. "
            f"Expected the cached blob from {DEFAULT_SOURCE_URL}."
        )
    try:
        return torch.load(viz_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(viz_path, map_location="cpu")


def _trim_zeros(sample: torch.Tensor) -> torch.Tensor:
    nz = (sample != 0).nonzero()
    if nz.numel() == 0:
        return sample[:0]
    return sample[: int(nz[-1]) + 1]


def _trim_to_last_snippet(text: str, enc: Encoding) -> str:
    parts = [part.strip("\n") for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not parts:
        raise ValueError("Decoded sample is empty after trimming.")
    last_block = parts[-1]
    lines = last_block.splitlines()

    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def _slice_from(index: int) -> str:
        return "\n".join(lines[index:]).strip("\n")

    def _token_len(snippet_lines: list[str]) -> int:
        return len(enc.encode("\n".join(snippet_lines).strip("\n")))

    def _drop_leading_comments(snippet_lines: list[str]) -> list[str]:
        trimmed = list(snippet_lines)
        while (
            len(trimmed) > 1
            and trimmed[0].strip().startswith("#")
            and _token_len(trimmed) > TARGET_SNIPPET_TOKENS
        ):
            trimmed = trimmed[1:]
        return trimmed

    def _select_short_suffix(snippet_lines: list[str]) -> list[str]:
        non_comment_indents = [
            _indent(line)
            for line in snippet_lines
            if line.strip() and not line.strip().startswith("#")
        ]
        base_indent = min(non_comment_indents) if non_comment_indents else 0

        def _starter_score(line: str) -> int:
            stripped = line.strip()
            if not stripped:
                return -1
            if stripped.startswith(
                (
                    "# Main",
                    "def ",
                    "class ",
                    "if ",
                    "elif ",
                    "else:",
                    "for ",
                    "while ",
                    "try:",
                    "except",
                    "with ",
                )
            ):
                return 3
            if stripped.startswith(("raise ", "return ", "print(")) or stripped.endswith(":"):
                return 2
            return 1

        candidates: list[tuple[int, int, list[str]]] = []
        for i, line in enumerate(snippet_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _indent(line) != base_indent:
                continue
            candidate = snippet_lines[i:]
            candidate_len = _token_len(candidate)
            if candidate_len <= TARGET_SNIPPET_TOKENS:
                candidates.append((_starter_score(line), candidate_len, candidate))
        if candidates:
            candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            return candidates[0][2]
        return snippet_lines

    for i, line in enumerate(lines):
        if line.strip() == "# Main":
            snippet_lines = _slice_from(i).splitlines()
            snippet_lines = _drop_leading_comments(snippet_lines)
            snippet_lines = _select_short_suffix(snippet_lines)
            return "\n".join(snippet_lines).strip("\n")

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if _indent(lines[i]) == 0 and (stripped.startswith("def ") or stripped.startswith("class ")):
            snippet_lines = _slice_from(i).splitlines()
            snippet_lines = _drop_leading_comments(snippet_lines)
            snippet_lines = _select_short_suffix(snippet_lines)
            return "\n".join(snippet_lines).strip("\n")

    snippet_lines = _drop_leading_comments(lines)
    snippet_lines = _select_short_suffix(snippet_lines)
    return "\n".join(snippet_lines).strip("\n")


def _normalize_for_pair_check(text: str) -> str:
    return text.replace('"', "'")


def _ensure_single_token(enc: Encoding, text: str) -> int:
    toks = enc.encode(text)
    if len(toks) != 1:
        raise ValueError(f"Expected {text!r} to be a single token, got {toks}.")
    return toks[0]


def _identifier_replacement_pattern(old: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(old)}\b")


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    rendered = text
    for old in sorted(replacements, key=len, reverse=True):
        new = replacements[old]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", old):
            rendered = _identifier_replacement_pattern(old).sub(new, rendered)
        else:
            rendered = rendered.replace(old, new)
    return rendered


def _placeholder_values(template_idx: int, replacements: dict[str, str]) -> dict[str, str]:
    placeholder_map = TEMPLATE_PLACEHOLDERS[template_idx]
    values = {}
    for source_text, rendered_value in replacements.items():
        if source_text not in placeholder_map:
            raise ValueError(
                f"Template {template_idx} replacement key {source_text!r} has no placeholder mapping."
            )
        values[placeholder_map[source_text]] = rendered_value
    return values


def _cross_product_dicts(slot_values: dict[str, list[str]], count: int) -> list[dict[str, str]]:
    keys = list(slot_values.keys())
    variants = []
    for values in product(*(slot_values[key] for key in keys)):
        variants.append(dict(zip(keys, values)))
        if len(variants) == count:
            return variants
    raise ValueError(f"Could not build {count} variants from slot values {list(slot_values.keys())}.")


def _variant_specs(variants_per_template: int) -> dict[int, list[dict[str, str]]]:
    if variants_per_template < 1:
        raise ValueError("variants_per_template must be positive.")

    distance_targets = [
        "node",
        "source",
        "start",
        "root",
        "origin",
        "entry",
        "pivot",
        "anchor",
        "hub",
        "base",
    ]
    robot_entities = [
        "Robot",
        "Worker",
        "Sensor",
        "Server",
        "Agent",
        "Thread",
        "Module",
        "Device",
        "Parser",
        "Solver",
    ]
    question_targets = [
        "Question 1",
        "Question 2",
        "Question 3",
        "Question 4",
        "Question 5",
        "Case 1",
        "Case 2",
        "Case 3",
        "Query 1",
        "Query 2",
    ]
    timestamp_vars = [
        "timestamp",
        "run_stamp",
        "log_stamp",
        "build_stamp",
        "save_stamp",
        "event_stamp",
        "export_stamp",
        "audit_stamp",
        "sync_stamp",
        "clock_stamp",
        "time_label",
        "run_label",
        "log_label",
        "build_label",
        "event_label",
        "audit_label",
        "save_label",
        "trace_label",
        "stamp_text",
        "time_token",
        "run_token",
        "log_token",
        "build_token",
        "event_token",
        "save_token",
        "clock_token",
        "trace_token",
        "export_token",
        "stamp_code",
        "time_code",
        "run_code",
        "log_code",
        "build_code",
        "event_code",
        "clock_code",
        "save_code",
        "audit_code",
        "trace_code",
        "sync_code",
        "export_code",
    ]
    project_vars = [
        "project",
        "workspace",
        "pipeline",
        "package",
        "release",
        "service",
        "module",
        "builder",
        "runner",
        "loader",
    ]
    project_titles = [
        "Project A",
        "Project B",
        "Project C",
        "Project D",
        "Project E",
        "Project F",
        "Project G",
        "Project H",
        "Project I",
        "Project J",
    ]
    clear_verbs = ["clear", "reset", "wipe", "flush", "clean"]
    clear_targets = [
        "grid",
        "board",
        "canvas",
        "layout",
        "panel",
        "matrix",
        "buffer",
        "screen",
    ]
    size_vars = [
        "n",
        "size",
        "count",
        "total",
        "limit",
        "width",
        "length",
        "max_n",
        "max_k",
        "nodes",
        "edges",
        "rows",
        "cols",
        "terms",
        "steps",
        "level",
        "slots",
        "files",
        "lines",
        "depth",
        "span",
        "items",
        "points",
        "ports",
        "bins",
        "pools",
        "cache",
        "frame",
        "block",
        "chunk",
        "batch",
        "index",
        "order",
        "range_n",
        "window",
        "groups",
        "parts",
        "tries",
        "stages",
        "samples",
        "retry_limit",
        "bucket",
        "degree_n",
    ]
    result_vars = [
        "results",
        "output",
        "answer",
        "record",
        "match",
        "event",
        "entry",
        "report",
        "issue",
        "alert",
        "values",
        "checks",
        "signals",
        "updates",
        "packets",
        "queries",
        "states",
        "frames",
        "rows",
        "cols",
        "samples",
        "groups",
        "items",
        "blocks",
        "chunks",
        "tasks",
        "jobs",
        "routes",
        "plans",
        "nodes",
        "edges",
        "paths",
        "tokens",
        "buffers",
        "layers",
        "windows",
        "ranges",
        "pairs",
        "logs",
        "notes",
    ]
    alt_vars = [
        "alt",
        "trial_cost",
        "next_cost",
        "path_cost",
        "new_cost",
        "best_guess",
        "route_cost",
        "candidate_cost",
        "scan_cost",
        "draft_cost",
    ]
    dist_vars = [
        "dist",
        "costs",
        "lengths",
        "best_cost",
        "min_cost",
        "path_map",
        "route_map",
        "score_map",
        "weight_map",
        "distance_map",
    ]
    node_vars = ["v", "node", "target", "neighbor", "vertex"]
    solution_vars = [
        "solution",
        "answer",
        "candidate",
        "match",
        "result",
        "route",
        "plan",
        "state",
        "path",
        "choice",
    ]
    solution_sets = [
        "solution_set",
        "answer_list",
        "candidate_pool",
        "match_list",
        "result_queue",
        "route_set",
        "plan_list",
        "state_list",
        "path_set",
        "choice_list",
    ]
    graph_vars = [
        "graph",
        "network",
        "tree",
        "mesh",
        "dag",
        "map_data",
        "flow_graph",
        "road_graph",
        "state_graph",
        "task_graph",
    ]
    graph_files = [
        "bigger_data.csv",
        "graph_data.csv",
        "network_data.csv",
        "mesh_data.csv",
        "tree_data.csv",
        "route_data.csv",
        "task_data.csv",
        "state_data.csv",
        "input_data.csv",
        "sample_data.csv",
    ]
    outer_items = [
        "vertex",
        "node",
        "row",
        "batch",
        "cluster",
        "record",
        "route",
        "segment",
        "group",
        "layer",
    ]
    inner_items = ["point", "entry", "item", "cell"]
    convert_functions = [
        "convert_to_set",
        "split_interval",
        "parse_range",
        "decode_span",
        "expand_window",
        "normalize_pair",
        "build_segment",
        "parse_bounds",
        "split_window",
        "decode_range",
    ]
    activity_args = [
        "activity",
        "segment",
        "interval",
        "window",
        "record",
        "span",
        "token",
        "payload",
        "entry",
        "range_text",
    ]
    pair_names = [
        ("start", "end"),
        ("left", "right"),
        ("lower", "upper"),
        ("begin", "finish"),
    ]

    specs = {
        0: [{"has_cycle": name} for name in [
            "has_cycle",
            "has_error",
            "has_conflict",
            "has_issue",
            "has_failure",
            "has_warning",
            "has_deadlock",
            "has_timeout",
            "has_overflow",
            "has_blocker",
            "loop_found",
            "error_found",
            "cycle_found",
            "fault_found",
            "retry_needed",
            "reset_needed",
            "input_broken",
            "graph_broken",
            "route_blocked",
            "state_invalid",
            "sort_failed",
            "check_failed",
            "merge_failed",
            "parse_failed",
            "write_failed",
            "read_failed",
            "link_missing",
            "edge_missing",
            "path_missing",
            "queue_stalled",
            "cache_stale",
            "stream_closed",
            "solver_stuck",
            "index_dirty",
            "job_blocked",
            "task_stalled",
            "update_pending",
            "cycle_pending",
            "alert_active",
            "guard_failed",
        ][:variants_per_template]],
        1: [
            {"Distance from node:": f"{prefix} from {target}."}
            for prefix, target in list(product(["Distance", "Cost", "Path", "Steps"], distance_targets))[
                :variants_per_template
            ]
        ],
        2: [
            {"Optimized Robot Usage:": f"{modifier} {entity} Usage."}
            for modifier, entity in list(
                product(["Optimized", "Balanced", "Tracked", "Planned"], robot_entities)
            )[:variants_per_template]
        ],
        3: [
            {"Output Question 1:": f"{prefix} {target}."}
            for prefix, target in list(
                product(["Output", "Result", "Prompt", "Check"], question_targets)
            )[:variants_per_template]
        ],
        4: [{"timestamp": name} for name in timestamp_vars[:variants_per_template]],
        5: _cross_product_dicts(
            {
                "project": project_vars,
                "Project A": project_titles,
            },
            variants_per_template,
        ),
        6: [
            {"Invalid encoded data": f"{adj} {noun}"}
            for adj, noun in list(
                product(
                    ["Invalid", "Malformed", "Broken", "Corrupted"],
                    [
                        "encoded data",
                        "token stream",
                        "input payload",
                        "byte sequence",
                        "record block",
                        "packet header",
                        "message body",
                        "config blob",
                        "state dump",
                        "archive chunk",
                    ],
                )
            )[:variants_per_template]
        ],
        7: _cross_product_dicts(
            {
                "clear_grid": [f"{verb}_{target}" for verb, target in product(clear_verbs, clear_targets)],
                "mine_file.txt": [
                    "mine_file.txt",
                    "grid_file.txt",
                    "board_file.txt",
                    "panel_file.txt",
                    "layout_file.txt",
                    "matrix_file.txt",
                    "buffer_file.txt",
                    "canvas_file.txt",
                    "screen_file.txt",
                    "level_file.txt",
                ],
            },
            variants_per_template,
        ),
        8: [{"n": name} for name in size_vars[:variants_per_template]],
        9: [{"results": name} for name in result_vars[:variants_per_template]],
        10: _cross_product_dicts(
            {
                "alt": alt_vars,
                "dist": dist_vars,
                "v": node_vars,
            },
            variants_per_template,
        ),
        11: _cross_product_dicts(
            {
                "solution": solution_vars,
                "solution_set": solution_sets,
            },
            variants_per_template,
        ),
        12: [
            {"No more unsatisfied points": f"no more {phrase}"}
            for phrase in [
                "unsatisfied points",
                "pending points",
                "open issues",
                "queued tasks",
                "active nodes",
                "remaining cells",
                "blocked entries",
                "stale records",
                "missing values",
                "failing checks",
                "unsolved cases",
                "waiting jobs",
                "pending routes",
                "open items",
                "dangling links",
                "empty slots",
                "invalid rows",
                "missing edges",
                "silent peers",
                "frozen workers",
                "stuck steps",
                "queued alerts",
                "stalled updates",
                "failing batches",
                "waiting queries",
                "broken states",
                "pending writes",
                "cached warnings",
                "running gaps",
                "orphaned tasks",
                "missing paths",
                "queued retries",
                "unmerged chunks",
                "waiting packets",
                "empty ranges",
                "stale snapshots",
                "missing markers",
                "open segments",
                "failing routes",
                "queued points",
            ][:variants_per_template]
        ],
        13: _cross_product_dicts(
            {
                "graph": graph_vars,
                "bigger_data.csv": graph_files,
            },
            variants_per_template,
        ),
        14: _cross_product_dicts(
            {
                "vertex": outer_items,
                "vertex_data": [f"{stem}_data" for stem in outer_items],
                "point": inner_items,
            },
            variants_per_template,
        ),
        15: _cross_product_dicts(
            {
                "convert_to_set": convert_functions,
                "activity": activity_args,
                "start": [pair[0] for pair in pair_names],
                "end": [pair[1] for pair in pair_names],
            },
            variants_per_template,
        ),
    }

    for template_idx, variants in specs.items():
        unique_variants = {tuple(sorted(variant.items())) for variant in variants}
        if len(unique_variants) < variants_per_template:
            raise ValueError(
                f"Template {template_idx} only has {len(unique_variants)} unique variants, "
                f"expected at least {variants_per_template}."
            )

    return specs


def _build_base_templates(enc: Encoding, viz_path: Path) -> list[dict[str, object]]:
    viz = _load_viz(viz_path)
    raw_task_samples = viz["importances"]["task_samples"]
    if not isinstance(raw_task_samples, tuple) or not raw_task_samples:
        raise ValueError("Unexpected task_samples payload; expected a non-empty tuple.")
    if not isinstance(raw_task_samples[0], torch.Tensor):
        raise ValueError("Expected task_samples[0] to be the tensor group of prompt samples.")

    group = raw_task_samples[0]
    if group.shape[0] % 2 != 0:
        raise ValueError(f"Expected an even number of task samples, got shape {tuple(group.shape)}.")

    decoded = [enc.decode(_trim_zeros(group[i]).tolist()) for i in range(group.shape[0])]
    half = group.shape[0] // 2
    base_rows = []
    for template_idx in range(half):
        first = decoded[template_idx]
        second = decoded[template_idx + half]
        if _normalize_for_pair_check(first) != _normalize_for_pair_check(second):
            raise ValueError(f"Sample pair {template_idx} does not differ only by quote style.")

        if first.count('"') >= second.count('"'):
            double_variant = first
            single_variant = second
        else:
            double_variant = second
            single_variant = first

        clean = _trim_to_last_snippet(single_variant, enc)
        corrupted = _trim_to_last_snippet(double_variant, enc)
        if len(enc.encode(clean)) != len(enc.encode(corrupted)):
            raise ValueError(
                f"Template {template_idx} has unequal token lengths after trimming: "
                f"{len(enc.encode(clean))} vs {len(enc.encode(corrupted))}."
            )
        base_rows.append(
            {
                "template_idx": template_idx,
                "template_label": TEMPLATE_LABELS[template_idx],
                "clean": clean,
                "corrupted": corrupted,
                "template_text": clean,
            }
        )

    for row in base_rows:
        override = MANUAL_TEMPLATE_OVERRIDES.get(int(row["template_idx"]))
        if not override:
            continue
        row["clean"] = override["clean"]
        row["corrupted"] = override["corrupted"]
        row["template_text"] = override["clean"]
        if len(enc.encode(str(row["clean"]))) != len(enc.encode(str(row["corrupted"]))):
            raise ValueError(
                f"Manual override for template {row['template_idx']} broke token-length parity."
            )

    for row in base_rows:
        template_idx = int(row["template_idx"])
        if template_idx not in TEMPLATE_PLACEHOLDERS:
            raise ValueError(f"Missing placeholder specification for template {template_idx}.")
        row["clean_template"] = _apply_replacements(str(row["clean"]), TEMPLATE_PLACEHOLDERS[template_idx])
        row["corrupted_template"] = _apply_replacements(
            str(row["corrupted"]), TEMPLATE_PLACEHOLDERS[template_idx]
        )
        row["template_text"] = row["clean_template"]

    return base_rows


def _expand_templates(enc: Encoding, base_rows: list[dict[str, object]], variants_per_template: int) -> pd.DataFrame:
    variant_specs = _variant_specs(variants_per_template)

    single_quote_paren_id = _ensure_single_token(enc, "')")
    double_quote_paren_id = _ensure_single_token(enc, '")')

    rows = []
    for base_row in base_rows:
        template_idx = int(base_row["template_idx"])
        clean_template = str(base_row["clean_template"])
        corrupted_template = str(base_row["corrupted_template"])
        template_text = str(base_row["template_text"])
        template_label = str(base_row["template_label"])

        replacements_list = variant_specs[template_idx]
        if len(replacements_list) < variants_per_template:
            raise ValueError(
                f"Template {template_idx} only has {len(replacements_list)} variants, "
                f"expected at least {variants_per_template}."
            )

        for variant_idx, replacements in enumerate(replacements_list[:variants_per_template]):
            placeholder_values = _placeholder_values(template_idx, replacements)
            single_clean = _apply_replacements(clean_template, placeholder_values)
            double_clean = _apply_replacements(corrupted_template, placeholder_values)

            single_len = len(enc.encode(single_clean))
            double_len = len(enc.encode(double_clean))
            if single_len != double_len:
                raise ValueError(
                    f"Template {template_idx} variant {variant_idx} has unequal token lengths: "
                    f"{single_len} vs {double_len}."
                )

            rows.append(
                {
                    "clean": single_clean,
                    "corrupted": double_clean,
                    "correct_answer": "')",
                    "correct_idx": single_quote_paren_id,
                    "incorrect_answer": '")',
                    "incorrect_idx": double_quote_paren_id,
                    "template_idx": template_idx,
                    "template_text": template_text,
                    "template_label": _template_label_for_quote_style(template_label, "single"),
                    "variant_idx": variant_idx,
                    "quote_style": "single",
                }
            )
            rows.append(
                {
                    "clean": double_clean,
                    "corrupted": single_clean,
                    "correct_answer": '")',
                    "correct_idx": double_quote_paren_id,
                    "incorrect_answer": "')",
                    "incorrect_idx": single_quote_paren_id,
                    "template_idx": template_idx,
                    "template_text": template_text,
                    "template_label": _template_label_for_quote_style(template_label, "double"),
                    "variant_idx": variant_idx,
                    "quote_style": "double",
                }
            )

    df = pd.DataFrame.from_records(rows)
    return df


def _sample_final_dataset(df: pd.DataFrame, num_examples: int, seed: int) -> pd.DataFrame:
    if num_examples % 2 != 0:
        raise ValueError("num_examples must be even so each naming variant keeps both quote directions.")

    unique_templates = sorted(df["template_idx"].unique().tolist())
    template_count = len(unique_templates)
    variants_needed = num_examples // 2
    base_variants = variants_needed // template_count
    remainder = variants_needed % template_count

    rng = pd.Series(unique_templates).sample(frac=1, random_state=seed).tolist()
    templates_with_extra_variant = set(rng[:remainder])

    selected_pairs = []
    for template_idx in unique_templates:
        quota = base_variants + (1 if template_idx in templates_with_extra_variant else 0)
        variant_pool = (
            df.loc[df["template_idx"] == template_idx, "variant_idx"]
            .drop_duplicates()
            .sample(frac=1, random_state=seed + template_idx)
            .tolist()
        )
        if quota > len(variant_pool):
            raise ValueError(
                f"Requested {quota} naming variants for template {template_idx}, "
                f"but only found {len(variant_pool)}."
            )
        for variant_idx in variant_pool[:quota]:
            selected_pairs.append((template_idx, variant_idx))

    selected_df = pd.DataFrame(selected_pairs, columns=["template_idx", "variant_idx"])
    final_df = df.merge(selected_df, on=["template_idx", "variant_idx"], how="inner")
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(final_df) != num_examples:
        raise ValueError(f"Expected {num_examples} final rows, got {len(final_df)}.")
    return final_df


def _assert_unique_samples(df: pd.DataFrame) -> None:
    duplicate_pairs = int(df.duplicated(subset=["clean", "corrupted"]).sum())
    if duplicate_pairs:
        raise ValueError(f"Found {duplicate_pairs} duplicate (clean, corrupted) pairs.")

    duplicate_clean = int(df.duplicated(subset=["clean"]).sum())
    if duplicate_clean:
        raise ValueError(f"Found {duplicate_clean} duplicate clean prompts.")

    duplicate_rows = int(
        df.duplicated(
            subset=[
                "clean",
                "corrupted",
                "template_idx",
                "template_label",
                "template_text",
                "variant_idx",
                "quote_style",
            ]
        ).sum()
    )
    if duplicate_rows:
        raise ValueError(f"Found {duplicate_rows} duplicate full rows.")


def main():
    args = parse_args()
    enc = _load_tinypython_encoding()
    base_rows = _build_base_templates(enc, args.viz_path)
    expanded_df = _expand_templates(enc, base_rows, args.variants_per_template)
    final_df = _sample_final_dataset(expanded_df, args.num_examples, args.seed)
    _assert_unique_samples(final_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(
        f"Wrote {len(final_df)} rows to {args.output} "
        f"from {args.variants_per_template} naming variants per template."
    )


if __name__ == "__main__":
    main()
