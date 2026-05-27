"""
Summarize safe Fusion PEFT result JSON files into a CSV table.
"""

import argparse
import csv
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import UNSEEN_RESULT_DIR  # noqa: E402


DEFAULT_PATTERNS = [
    "fusion_baseline_*.json",
    "fusion_adapter_visual_only_*.json",
    "fusion_lora_visual_only_*.json",
    "fusion_coop_seen_anchor_*.json",
    "fusion_cocoop_seen_anchor_*.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a CSV summary for safe Fusion PEFT retrieval results."
    )
    parser.add_argument("--result_dir", type=str, default=UNSEEN_RESULT_DIR)
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(UNSEEN_RESULT_DIR, "peft_summary.csv"),
    )
    parser.add_argument("--strict_safety", action="store_true")
    return parser.parse_args()


def collect_inputs(result_dir: str, inputs):
    if inputs:
        return inputs

    paths = []
    for pattern in DEFAULT_PATTERNS:
        paths.extend(glob.glob(os.path.join(result_dir, pattern)))
    return sorted(set(paths))


def load_result(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def safety_ok(data: dict) -> bool:
    return (
        data.get("uses_unseen_class_names_for_ranking") is False
        and data.get("uses_unseen_labels_for_ranking") is False
        and data.get("uses_gallery_labels_for_ranking") is False
        and data.get("unseen_labels_used_only_for_metrics") is True
    )


def make_row(path: str, data: dict):
    metrics = data.get("metrics", {})
    return {
        "file": os.path.basename(path),
        "method": data.get("method", ""),
        "ranking_score_type": data.get("ranking_score_type", ""),
        "feature_source": data.get("feature_source", ""),
        "text_anchor_scope": data.get("text_anchor_scope", ""),
        "seen_anchor_weight": data.get("seen_anchor_weight", ""),
        "seen_anchor_similarity": data.get("seen_anchor_similarity", ""),
        "seen_anchor_power": data.get("seen_anchor_power", ""),
        "mAP": metrics.get("mAP", ""),
        "NDCG@100": metrics.get("NDCG@100", metrics.get("NDCG", "")),
        "ANMRR": metrics.get("ANMRR", ""),
        "Recall@100": metrics.get("Recall@100", ""),
        "safety_ok": safety_ok(data),
    }


def write_csv_with_fallback(output_path: str, fieldnames, rows) -> str:
    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_path
    except PermissionError:
        parent_dir = os.path.dirname(os.path.dirname(output_path))
        fallback_path = os.path.join(parent_dir, os.path.basename(output_path))
        if fallback_path == output_path:
            raise
        with open(fallback_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"Warning: could not write to {output_path}; "
            f"saved fallback summary to {fallback_path}"
        )
        return fallback_path


def main():
    args = parse_args()
    paths = collect_inputs(args.result_dir, args.inputs)
    if not paths:
        raise RuntimeError("No Fusion PEFT result JSON files found.")

    rows = []
    unsafe_paths = []
    for path in paths:
        data = load_result(path)
        row = make_row(path, data)
        rows.append(row)
        if not row["safety_ok"]:
            unsafe_paths.append(path)

    if args.strict_safety and unsafe_paths:
        joined = "\n".join(unsafe_paths)
        raise RuntimeError(f"Safety metadata check failed for:\n{joined}")

    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    args.output = write_csv_with_fallback(args.output, fieldnames, rows)

    print(f"Saved summary: {args.output}")
    print(f"Rows: {len(rows)}")
    if unsafe_paths:
        print("Warning: some files failed the safety metadata check:")
        for path in unsafe_paths:
            print(f"  {path}")
    else:
        print("Safety metadata check passed for all summarized files.")


if __name__ == "__main__":
    main()
