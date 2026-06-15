import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd
from tiktoken import Encoding


DEFAULT_OUTPUT_PATH = Path(__file__).with_name("tinypython_2k.csv")
DEFAULT_SOURCE_URL = (
    "https://openaipublic.blob.core.windows.net/circuit-sparsity/viz/"
    "csp_yolo2/else_elif_beeg/prune_v4/k_optim/viz_data.pt"
)
DEFAULT_NUM_EXAMPLES = 1000
DEFAULT_VARIANTS_PER_TEMPLATE = 40

BASE_TEMPLATES = [
    {
        "template_idx": 0,
        "template_label": "return_result",
        "clean_template": "if [RESULT_VAR]:\n    [CACHE_VAR] = [RESULT_VAR]\n    return [RESULT_VAR]\nelse",
        "corrupted_template": "if [RESULT_VAR]:\n    [CACHE_VAR] = [RESULT_VAR]\n    return [RESULT_VAR]\nelif",
    },
    {
        "template_idx": 1,
        "template_label": "answer_check",
        "clean_template": "if eval([PROBLEM_VAR]) == [ANSWER_VAR]:\n    print([SUCCESS_MSG])\nelse",
        "corrupted_template": "if eval([PROBLEM_VAR]) == [ANSWER_VAR]:\n    print([SUCCESS_MSG])\nelif",
    },
    {
        "template_idx": 2,
        "template_label": "append_guard",
        "clean_template": "if [COUNT_MAP][[NODE_VAR].topic] >= [TARGET_VAR]:\n    [REDIST_LIST].append([NODE_VAR])\nelse",
        "corrupted_template": "if [COUNT_MAP][[NODE_VAR].topic] >= [TARGET_VAR]:\n    [REDIST_LIST].append([NODE_VAR])\nelif",
    },
    {
        "template_idx": 3,
        "template_label": "program_dir_chain",
        "clean_template": "if [DIR_VAR] == 0:\n    [DIR_VAR] = [NEG_VAR]\nelif [DIR_VAR] == 1:\n    [DIR_VAR] = [POS_VAR]\nelse",
        "corrupted_template": "if [DIR_VAR] == 0:\n    [DIR_VAR] = [NEG_VAR]\nelif [DIR_VAR] == 1:\n    [DIR_VAR] = [POS_VAR]\nelif",
    },
    {
        "template_idx": 4,
        "template_label": "schedule_branch",
        "clean_template": "if [TIME_VAR] > [LAST_VAR]:\n    [START_VAR] = [LAST_VAR] + 1\nelse",
        "corrupted_template": "if [TIME_VAR] > [LAST_VAR]:\n    [START_VAR] = [LAST_VAR] + 1\nelif",
    },
    {
        "template_idx": 5,
        "template_label": "strategy_branch",
        "clean_template": "if [STRATEGY_VAR] == 'bfs':\n    return bfs([START_VAR], [VISITED_VAR], [GOAL_VAR], [NEXT_VAR])\nelif [STRATEGY_VAR] == 'dfs':\n    return dfs([START_VAR], [VISITED_VAR], [GOAL_VAR], [NEXT_VAR])\nelse",
        "corrupted_template": "if [STRATEGY_VAR] == 'bfs':\n    return bfs([START_VAR], [VISITED_VAR], [GOAL_VAR], [NEXT_VAR])\nelif [STRATEGY_VAR] == 'dfs':\n    return dfs([START_VAR], [VISITED_VAR], [GOAL_VAR], [NEXT_VAR])\nelif",
    },
    {
        "template_idx": 6,
        "template_label": "equality_case",
        "clean_template": "if [CONSTRAINT_VAR].is_equality():\n    return [Z3_VAR] == -[COEFF_VAR]['1']\nelse",
        "corrupted_template": "if [CONSTRAINT_VAR].is_equality():\n    return [Z3_VAR] == -[COEFF_VAR]['1']\nelif",
    },
    {
        "template_idx": 7,
        "template_label": "stats_update",
        "clean_template": "if [ROOT_VAR] not in [STATS_VAR]:\n    [STATS_VAR][[ROOT_VAR]] = ([AREA_VAR], [COUNT_VAR])\nelse",
        "corrupted_template": "if [ROOT_VAR] not in [STATS_VAR]:\n    [STATS_VAR][[ROOT_VAR]] = ([AREA_VAR], [COUNT_VAR])\nelif",
    },
    {
        "template_idx": 8,
        "template_label": "inline_threshold",
        "clean_template": "if ([NUM_VAR] <= [LIMIT]):\n    return [FUNC_CALL]\nelse",
        "corrupted_template": "if ([NUM_VAR] <= [LIMIT]):\n    return [FUNC_CALL]\nelif",
    },
    {
        "template_idx": 9,
        "template_label": "string_align",
        "clean_template": "if [CHAR_VAR] in [STRING_VAR]:\n    [LEFT_VAR], [RIGHT_VAR] = [STRING_VAR].split([CHAR_VAR], 1)\nelse",
        "corrupted_template": "if [CHAR_VAR] in [STRING_VAR]:\n    [LEFT_VAR], [RIGHT_VAR] = [STRING_VAR].split([CHAR_VAR], 1)\nelif",
    },
    {
        "template_idx": 10,
        "template_label": "binary_search",
        "clean_template": "if [DATA_VAR][[INDEX_VAR]] < [VALUE_VAR]:\n    [RESULT_VAR] = [INDEX_VAR]\n    [LOW_VAR] = [INDEX_VAR] + 1\nelse",
        "corrupted_template": "if [DATA_VAR][[INDEX_VAR]] < [VALUE_VAR]:\n    [RESULT_VAR] = [INDEX_VAR]\n    [LOW_VAR] = [INDEX_VAR] + 1\nelif",
    },
    {
        "template_idx": 11,
        "template_label": "truthy_call",
        "clean_template": "if [ROOM_VAR]:\n    set_parameter([BOARD_VAR], [PARAM_VAR], [ROOM_NAME] + ' - ' + [ROOM_VAR])\nelse",
        "corrupted_template": "if [ROOM_VAR]:\n    set_parameter([BOARD_VAR], [PARAM_VAR], [ROOM_NAME] + ' - ' + [ROOM_VAR])\nelif",
    },
    {
        "template_idx": 12,
        "template_label": "compare_objects",
        "clean_template": "if isinstance([LEFT_VAR], (list, tuple)):\n    if isinstance([RIGHT_VAR], (list, tuple)):\n        compare_objects([LEFT_VAR][0], [RIGHT_VAR][0], [NAME_VAR], [PATH_VAR])\n    else",
        "corrupted_template": "if isinstance([LEFT_VAR], (list, tuple)):\n    if isinstance([RIGHT_VAR], (list, tuple)):\n        compare_objects([LEFT_VAR][0], [RIGHT_VAR][0], [NAME_VAR], [PATH_VAR])\n    elif",
    },
    {
        "template_idx": 13,
        "template_label": "cell_toggle",
        "clean_template": "if [BOARD_VAR][[ROW_VAR]][[COL_VAR]] == [OFF_VAR]:\n    [BOARD_VAR][[ROW_VAR]][[COL_VAR]] = 1\nelse",
        "corrupted_template": "if [BOARD_VAR][[ROW_VAR]][[COL_VAR]] == [OFF_VAR]:\n    [BOARD_VAR][[ROW_VAR]][[COL_VAR]] = 1\nelif",
    },
    {
        "template_idx": 14,
        "template_label": "random_split",
        "clean_template": "if random.random() < 0.5:\n    [OUT_VAR].append(f\"ML,{[PEER_VAR]},{[ELEMENT_VAR]}\")\nelse",
        "corrupted_template": "if random.random() < 0.5:\n    [OUT_VAR].append(f\"ML,{[PEER_VAR]},{[ELEMENT_VAR]}\")\nelif",
    },
    {
        "template_idx": 15,
        "template_label": "none_default",
        "clean_template": "if [END_VAR] is None:\n    [GOAL_VAR] = ([ROWS_VAR]-1, [COLS_VAR]-1)\nelse",
        "corrupted_template": "if [END_VAR] is None:\n    [GOAL_VAR] = ([ROWS_VAR]-1, [COLS_VAR]-1)\nelif",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deterministic else-vs-elif data for sparse transformer experiments."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling.")
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
        help="How many substitution variants to create for each base template before sampling.",
    )
    return parser.parse_args()


def _template_label_for_branch_style(base_label: str, branch_style: str) -> str:
    if branch_style == "else":
        return f"{base_label}_else"
    if branch_style == "elif":
        return f"{base_label}_elif"
    raise ValueError(f"Unexpected branch_style: {branch_style!r}")


def _load_tinypython_encoding() -> Encoding:
    repo_root = Path(__file__).resolve().parents[3] / "circuit_sparsity"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from circuit_sparsity.tiktoken_ext import tinypython

    return Encoding(**tinypython.tinypython_2k())


def _render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _cross_product_dicts(slot_values: dict[str, list[str]], count: int) -> list[dict[str, str]]:
    keys = list(slot_values.keys())
    variants = []
    for values in product(*(slot_values[key] for key in keys)):
        variants.append(dict(zip(keys, values)))
        if len(variants) == count:
            return variants
    raise ValueError(f"Could not build {count} variants from slot values {list(slot_values.keys())}.")


def _variant_specs(variants_per_template: int) -> dict[int, list[dict[str, str]]]:
    result_vars = ["result", "value", "output", "status", "match", "response", "signal", "payload"]
    cache_vars = ["cache", "saved", "buffer", "record", "entry", "memo", "slot", "note"]
    answer_vars = ["problem", "expr", "query", "prompt", "formula", "statement", "clause", "rule"]
    answer_names = ["answer", "guess", "reply", "target", "prediction", "solution", "label", "check"]
    topic_maps = ["label_counts", "topic_counts", "group_counts", "bucket_counts", "node_counts", "quota_map", "limit_map", "track_counts"]
    redist_lists = ["redistribute_nodes", "carry_nodes", "spill_nodes", "retry_nodes", "extra_nodes", "moved_nodes", "defer_nodes", "shift_nodes"]
    node_vars = ["node", "item", "entry", "candidate", "record", "sample", "value", "packet"]
    dir_vars = ["program_dir", "flow_dir", "route_dir", "step_dir", "turn_dir", "scan_dir", "axis_dir", "move_dir"]
    neg_vars = ["reverse_step", "left_turn", "back_dir", "prev_dir", "neg_dir", "downstream", "backward", "flip_dir"]
    pos_vars = ["forward_step", "right_turn", "next_dir", "lead_dir", "pos_dir", "upstream", "advance", "keep_dir"]
    time_vars = ["time", "eta", "start_at", "arrival", "slot_time", "target_time", "next_time", "ready_time"]
    last_vars = ["last_train_time", "last_seen", "prev_time", "last_time", "stored_time", "recent_time", "edge_time", "stop_time"]
    start_vars = ["start_time", "begin_at", "offset_time", "launch_time", "open_time", "queue_time", "entry_time", "resume_time"]
    strategy_vars = ["strategy", "mode", "search_mode", "plan_mode", "route_mode", "walk_mode", "pick_mode", "scan_mode"]
    constraint_vars = ["isl_constraint", "shape_constraint", "poly_constraint", "line_constraint", "rule_constraint", "grid_constraint", "bound_constraint", "tile_constraint"]
    z3_vars = ["z3_constraint", "expr_total", "solver_expr", "constraint_sum", "check_expr", "rule_expr", "shape_expr", "bound_expr"]
    coeff_vars = ["coefficients", "weights", "factors", "terms", "params", "coefs", "sizes", "limits"]
    root_vars = ["root_id", "root_key", "origin_id", "base_id", "anchor_id", "group_id", "family_id", "house_id"]
    stats_vars = ["family_stats", "root_stats", "area_stats", "household_stats", "group_stats", "tree_stats", "cluster_stats", "region_stats"]
    area_vars = ["area", "zone", "span", "size", "weight", "reach", "width", "field"]
    count_vars = ["household_count", "member_count", "group_count", "sample_count", "node_count", "row_count", "hit_count", "unit_count"]
    num_vars = ["nums", "count", "total", "value", "score", "points", "size", "level"]
    func_calls = ["gianlyung(nums, value)", "solver(count, total)", "reduce_score(total, value)", "walk_path(size, count)", "rank_nodes(value, total)"]
    char_vars = ["character", "delimiter", "marker", "pivot", "splitter", "token_char", "break_char", "join_char"]
    string_vars = ["string", "row_text", "line_text", "entry_text", "source_text", "raw_text", "block_text", "name_text"]
    left_vars = ["left", "lhs", "prefix", "head", "front", "begin", "lead", "first"]
    right_vars = ["right", "rhs", "suffix", "tail", "back", "rest", "remain", "second"]
    data_vars = ["data", "values", "scores", "items", "records", "points", "levels", "nodes"]
    index_vars = ["index", "mid", "pivot", "probe", "cursor", "ptr", "slot", "step"]
    value_vars = ["value", "target", "threshold", "probe_value", "goal", "needle", "limit", "pivot_value"]
    result_slots = ["result", "best_idx", "last_idx", "match_idx", "stop_idx", "slot_idx", "found_idx", "chosen_idx"]
    low_vars = ["low", "start", "left_idx", "begin", "min_idx", "floor_idx", "scan_low", "range_low"]
    room_vars = ["room_num", "room_id", "room_code", "room_tag", "unit_num", "unit_code", "slot_num", "cell_num"]
    board_vars = ["board_instance", "board_obj", "panel", "unit", "device", "segment", "record", "item"]
    param_vars = ["param_key", "room_param", "field_param", "target_param", "slot_param", "ref_param", "code_param", "name_param"]
    room_names = ["room_name", "unit_name", "cell_name", "slot_name", "zone_name", "panel_name", "board_name", "space_name"]
    compare_vars = ["left", "right", "lhs", "rhs", "first_obj", "second_obj", "source_obj", "target_obj"]
    name_vars = ["name", "label", "title", "tag", "key", "kind", "group", "shape"]
    path_vars = ["path", "trace", "route", "chain", "cursor", "trail", "link", "branch"]
    board_arrays = ["arr", "cells", "grid", "board", "matrix", "slots", "tiles", "marks"]
    row_vars = ["y", "row", "r", "row_idx", "iy", "line", "top", "h"]
    col_vars = ["x", "col", "c", "col_idx", "ix", "column", "left", "w"]
    off_vars = ["self.C[0]", "dead_cell", "blank_cell", "off_cell", "base_cell"]
    out_vars = ["constraint_list", "output", "events", "rules", "rows", "moves", "pairs", "results"]
    peer_vars = ["peer", "other", "partner", "neighbor", "mate", "link", "pair", "ally"]
    element_vars = ["element", "item", "value", "node", "entry", "sample", "token", "term"]
    end_vars = ["end", "goal", "finish", "stop", "target", "exit_node", "limit_end", "final_pos"]
    goal_vars = ["goal", "finish", "target", "end_pos", "stop_pos", "exit_pos", "goal_node", "final_state"]
    rows_vars = ["self.num_rows", "rows", "row_count", "grid_rows", "maze_rows"]
    cols_vars = ["self.num_cols", "cols", "col_count", "grid_cols", "maze_cols"]

    return {
        0: _cross_product_dicts({"[RESULT_VAR]": result_vars, "[CACHE_VAR]": cache_vars}, variants_per_template),
        1: _cross_product_dicts({"[PROBLEM_VAR]": answer_vars, "[ANSWER_VAR]": answer_names, "[SUCCESS_MSG]": ['"ok"', '"match"', '"success"', '"passed"', '"clear"']}, variants_per_template),
        2: _cross_product_dicts({"[COUNT_MAP]": topic_maps, "[TARGET_VAR]": ["target_size", "limit_size", "goal_size", "quota_size", "cap_size"], "[REDIST_LIST]": redist_lists, "[NODE_VAR]": node_vars}, variants_per_template),
        3: _cross_product_dicts({"[DIR_VAR]": dir_vars, "[NEG_VAR]": neg_vars, "[POS_VAR]": pos_vars}, variants_per_template),
        4: _cross_product_dicts({"[TIME_VAR]": time_vars, "[LAST_VAR]": last_vars, "[START_VAR]": start_vars}, variants_per_template),
        5: _cross_product_dicts({"[STRATEGY_VAR]": strategy_vars, "[START_VAR]": ["start", "source", "origin", "root", "entry"], "[VISITED_VAR]": ["visited", "seen", "marked", "closed", "used"], "[GOAL_VAR]": ["goal_test", "is_goal", "target_test", "hit_goal", "done_test"], "[NEXT_VAR]": ["successors", "neighbors", "next_nodes", "outgoing", "steps"]}, variants_per_template),
        6: _cross_product_dicts({"[CONSTRAINT_VAR]": constraint_vars, "[Z3_VAR]": z3_vars, "[COEFF_VAR]": coeff_vars}, variants_per_template),
        7: _cross_product_dicts({"[ROOT_VAR]": root_vars, "[STATS_VAR]": stats_vars, "[AREA_VAR]": area_vars, "[COUNT_VAR]": count_vars}, variants_per_template),
        8: _cross_product_dicts({"[NUM_VAR]": num_vars, "[LIMIT]": ["15", "20", "24", "30", "32"], "[FUNC_CALL]": func_calls}, variants_per_template),
        9: _cross_product_dicts({"[CHAR_VAR]": char_vars, "[STRING_VAR]": string_vars, "[LEFT_VAR]": left_vars, "[RIGHT_VAR]": right_vars}, variants_per_template),
        10: _cross_product_dicts({"[DATA_VAR]": data_vars, "[INDEX_VAR]": index_vars, "[VALUE_VAR]": value_vars, "[RESULT_VAR]": result_slots, "[LOW_VAR]": low_vars}, variants_per_template),
        11: _cross_product_dicts({"[ROOM_VAR]": room_vars, "[BOARD_VAR]": board_vars, "[PARAM_VAR]": param_vars, "[ROOM_NAME]": room_names}, variants_per_template),
        12: _cross_product_dicts({"[LEFT_VAR]": compare_vars, "[RIGHT_VAR]": compare_vars[::-1], "[NAME_VAR]": name_vars, "[PATH_VAR]": path_vars}, variants_per_template),
        13: _cross_product_dicts({"[BOARD_VAR]": board_arrays, "[ROW_VAR]": row_vars, "[COL_VAR]": col_vars, "[OFF_VAR]": off_vars}, variants_per_template),
        14: _cross_product_dicts({"[OUT_VAR]": out_vars, "[PEER_VAR]": peer_vars, "[ELEMENT_VAR]": element_vars}, variants_per_template),
        15: _cross_product_dicts({"[END_VAR]": end_vars, "[GOAL_VAR]": goal_vars, "[ROWS_VAR]": rows_vars, "[COLS_VAR]": cols_vars}, variants_per_template),
    }


def _expand_templates(enc: Encoding, base_rows: list[dict[str, object]], variants_per_template: int) -> pd.DataFrame:
    variant_specs = _variant_specs(variants_per_template)
    rows = []
    for base_row in base_rows:
        template_idx = int(base_row["template_idx"])
        clean_template = str(base_row["clean_template"])
        corrupted_template = str(base_row["corrupted_template"])
        template_text = clean_template
        template_label = str(base_row["template_label"])

        for variant_idx, replacements in enumerate(variant_specs[template_idx]):
            else_clean = _render_template(clean_template, replacements)
            elif_clean = _render_template(corrupted_template, replacements)
            if len(else_clean) == 0 or len(elif_clean) == 0:
                raise ValueError("Rendered prompt is empty.")
            else_len = len(enc.encode(else_clean))
            elif_len = len(enc.encode(elif_clean))
            if else_len != elif_len:
                raise ValueError(
                    f"Template {template_idx} variant {variant_idx} has unequal token lengths: "
                    f"{else_len} vs {elif_len}."
                )
            rows.append(
                {
                    "clean": else_clean,
                    "corrupted": elif_clean,
                    "colon_newline": 1,
                    "template_idx": template_idx,
                    "template_text": template_text,
                    "template_label": _template_label_for_branch_style(template_label, "else"),
                    "variant_idx": variant_idx,
                    "branch_style": "else",
                }
            )
            rows.append(
                {
                    "clean": elif_clean,
                    "corrupted": else_clean,
                    "colon_newline": 0,
                    "template_idx": template_idx,
                    "template_text": template_text,
                    "template_label": _template_label_for_branch_style(template_label, "elif"),
                    "variant_idx": variant_idx,
                    "branch_style": "elif",
                }
            )
    return pd.DataFrame.from_records(rows)


def _sample_final_dataset(df: pd.DataFrame, num_examples: int, seed: int) -> pd.DataFrame:
    if num_examples % 2 != 0:
        raise ValueError("num_examples must be even so each naming variant keeps both branch directions.")

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
                "colon_newline",
                "template_idx",
                "template_text",
                "template_label",
                "variant_idx",
                "branch_style",
            ]
        ).sum()
    )
    if duplicate_rows:
        raise ValueError(f"Found {duplicate_rows} duplicate full rows.")


def main():
    args = parse_args()
    enc = _load_tinypython_encoding()
    expanded_df = _expand_templates(enc, BASE_TEMPLATES, args.variants_per_template)
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
