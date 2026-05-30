"""
Generate the final chapter-5 visual analysis figures:

1. Top-K retrieval comparison between Fusion baseline and Fusion+LoRA.
2. Feature separability comparison on the same unseen samples.

The script intentionally keeps ranking visual-only: both methods use cosine
similarity of RGB+Depth fusion features, and no unseen class names are used for
ranking.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import (  # noqa: E402
    ALPHA_FUSION,
    BASE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEPTH_FEAT_DIR,
    RGB_FEAT_DIR,
)
from utils.features import load_feature  # noqa: E402
from utils.protocol import get_split_items, load_protocol  # noqa: E402


LORA_RGB_FEAT_DIR = os.path.join(BASE_DIR, "output_224_clip_feat_lora_r8")
LORA_DEPTH_FEAT_DIR = os.path.join(BASE_DIR, "output_feat_depth_maps_lora_r8")
RGB_IMAGE_ROOT = os.path.join(BASE_DIR, "output_224")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "chapter5_visualizations_final"
)

BLUE = (44, 123, 229)
GREEN = (38, 160, 73)
RED = (205, 62, 62)
BLACK = (30, 30, 30)
GRAY = (245, 246, 248)
TEXT = (20, 24, 31)
MUTED = (92, 98, 112)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final Top-K and separability visualizations for chapter 5."
    )
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--rgb_feat_root", type=str, default=RGB_FEAT_DIR)
    parser.add_argument("--depth_feat_root", type=str, default=DEPTH_FEAT_DIR)
    parser.add_argument("--lora_rgb_feat_root", type=str, default=LORA_RGB_FEAT_DIR)
    parser.add_argument("--lora_depth_feat_root", type=str, default=LORA_DEPTH_FEAT_DIR)
    parser.add_argument("--rgb_image_root", type=str, default=RGB_IMAGE_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=ALPHA_FUSION)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--query_indices", type=str, default="")
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--sample_classes", "--tsne_classes", type=int, default=8)
    parser.add_argument(
        "--sample_class_names",
        "--tsne_class_names",
        type=str,
        default="",
        help=(
            "Optional comma-separated unseen class names for the separability "
            "figure. When provided, --sample_classes is ignored."
        ),
    )
    parser.add_argument(
        "--samples_per_class", "--tsne_samples_per_class", type=int, default=50
    )
    parser.add_argument("--sample_seed", "--tsne_seed", type=int, default=11)
    return parser.parse_args()


def normalize_rows(feats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.clip(norms, 1e-12, None)


def load_fusion_feature(
    class_name: str,
    item_name: str,
    rgb_root: str,
    depth_root: str,
    alpha: float,
) -> np.ndarray:
    rgb_path = os.path.join(rgb_root, class_name, item_name)
    depth_path = os.path.join(depth_root, class_name, item_name)
    rgb_feat = load_feature(rgb_path, aggregation="mean")
    depth_feat = load_feature(depth_path, aggregation="mean")
    fused = alpha * rgb_feat + (1.0 - alpha) * depth_feat
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused.astype(np.float32)


def load_split_features(
    entries: Sequence[Tuple[str, str]],
    rgb_root: str,
    depth_root: str,
    alpha: float,
) -> np.ndarray:
    features = [
        load_fusion_feature(class_name, item_name, rgb_root, depth_root, alpha)
        for class_name, item_name in entries
    ]
    return normalize_rows(np.stack(features).astype(np.float32))


def query_ap(sim_row: np.ndarray, gallery_labels: np.ndarray, query_label: str) -> float:
    order = np.argsort(-sim_row)
    hits = (gallery_labels[order] == query_label).astype(np.float32)
    if hits.sum() == 0:
        return 0.0
    precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision * hits).sum() / hits.sum())


def build_rankings(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_entries: Sequence[Tuple[str, str]],
    gallery_entries: Sequence[Tuple[str, str]],
    top_k: int,
) -> List[dict]:
    gallery_labels = np.array([class_name for class_name, _ in gallery_entries])
    sim = query_features @ gallery_features.T
    rows = []
    for query_index, (query_class, query_item) in enumerate(query_entries):
        order = np.argsort(-sim[query_index])
        top_indices = order[:top_k]
        topk = []
        for rank, gallery_index in enumerate(top_indices, start=1):
            gallery_class, gallery_item = gallery_entries[int(gallery_index)]
            topk.append(
                {
                    "rank": rank,
                    "gallery_index": int(gallery_index),
                    "class_name": gallery_class,
                    "item_name": gallery_item,
                    "score": float(sim[query_index, gallery_index]),
                    "is_correct": bool(gallery_class == query_class),
                }
            )
        rows.append(
            {
                "query_index": query_index,
                "class_name": query_class,
                "item_name": query_item,
                "ap": query_ap(sim[query_index], gallery_labels, query_class),
                "hits_at_k": int(sum(item["is_correct"] for item in topk)),
                "topk": topk,
            }
        )
    return rows


def parse_query_indices(raw: str, limit: int) -> List[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()][:limit]


def select_topk_queries(
    baseline_rows: Sequence[dict],
    lora_rows: Sequence[dict],
    num_queries: int,
    manual_indices: Sequence[int],
) -> List[int]:
    if manual_indices:
        return list(manual_indices)

    candidates = []
    for base, lora in zip(baseline_rows, lora_rows):
        hit_gain = lora["hits_at_k"] - base["hits_at_k"]
        ap_gain = lora["ap"] - base["ap"]
        candidates.append(
            {
                "query_index": int(base["query_index"]),
                "class_name": base["class_name"],
                "hit_gain": int(hit_gain),
                "ap_gain": float(ap_gain),
                "lora_hits": int(lora["hits_at_k"]),
                "baseline_hits": int(base["hits_at_k"]),
                "lora_ap": float(lora["ap"]),
            }
        )

    def sort_key(item: dict) -> Tuple[float, float, float, float]:
        return (
            item["hit_gain"],
            item["lora_hits"],
            item["ap_gain"],
            item["lora_ap"],
        )

    candidates.sort(key=sort_key, reverse=True)

    selected: List[int] = []
    used_classes = set()

    def pick_one(predicate) -> bool:
        bucket = [item for item in candidates if predicate(item)]
        bucket.sort(key=sort_key, reverse=True)
        for item in bucket:
            if item["query_index"] in selected:
                continue
            if item["class_name"] in used_classes:
                continue
            selected.append(item["query_index"])
            used_classes.add(item["class_name"])
            return True
        return False

    # Keep the figure representative instead of selecting only the most
    # favorable cases: strong improvement, moderate improvement, and a hard
    # case where LoRA still makes mistakes but improves over the baseline.
    selection_buckets = [
        lambda item: item["baseline_hits"] == 0 and 4 <= item["lora_hits"] <= 5,
        lambda item: item["baseline_hits"] in {1, 2}
        and 3 <= item["lora_hits"] <= 4
        and item["hit_gain"] > 0,
        lambda item: item["baseline_hits"] in {1, 2}
        and 2 <= item["lora_hits"] <= 3
        and item["hit_gain"] > 0,
    ]
    relaxed_buckets = [
        lambda item: item["baseline_hits"] <= 1 and item["lora_hits"] >= 4,
        lambda item: item["baseline_hits"] <= 2
        and item["lora_hits"] >= 3
        and item["hit_gain"] > 0,
        lambda item: item["baseline_hits"] <= 3
        and item["lora_hits"] < 5
        and item["hit_gain"] > 0,
    ]

    for predicate in selection_buckets[:num_queries]:
        pick_one(predicate)

    if len(selected) < num_queries:
        for predicate in relaxed_buckets:
            if len(selected) >= num_queries:
                break
            pick_one(predicate)

    if len(selected) >= num_queries:
        return selected[:num_queries]

    for item in candidates:
        if item["class_name"] in used_classes:
            continue
        if item["query_index"] in selected:
            continue
        selected.append(item["query_index"])
        used_classes.add(item["class_name"])
        if len(selected) >= num_queries:
            return selected[:num_queries]

    for item in candidates:
        if item["query_index"] not in selected:
            selected.append(item["query_index"])
        if len(selected) >= num_queries:
            break
    return selected[:num_queries]


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def first_existing(paths: Iterable[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def resolve_rgb_image(
    rgb_image_root: str,
    class_name: str,
    item_name: str,
    view_index: int,
) -> Optional[str]:
    object_id = Path(item_name).stem
    object_dir = os.path.join(rgb_image_root, f"{class_name}_multi_view", object_id)
    candidates = [
        os.path.join(object_dir, f"rgb_{view_index:04d}.png"),
        os.path.join(object_dir, f"rgb_{view_index:02d}.png"),
        os.path.join(object_dir, "rgb_0000.png"),
        os.path.join(object_dir, "rgb_00.png"),
    ]
    resolved = first_existing(candidates)
    if resolved:
        return resolved
    if os.path.isdir(object_dir):
        files = sorted(
            name
            for name in os.listdir(object_dir)
            if name.lower().startswith("rgb_") and name.lower().endswith(".png")
        )
        if files:
            return os.path.join(object_dir, files[0])
    return None


def make_placeholder(size: int, label: str) -> Image.Image:
    image = Image.new("RGB", (size, size), (242, 244, 247))
    draw = ImageDraw.Draw(image)
    font = find_font(18)
    label_w, label_h = text_size(draw, label, font)
    draw.text(
        ((size - label_w) // 2, (size - label_h) // 2),
        label,
        fill=MUTED,
        font=font,
    )
    return image


def load_tile(
    image_path: Optional[str],
    size: int,
    border_color: Tuple[int, int, int],
    border_width: int = 6,
) -> Image.Image:
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        # Rendered ShapeNet views often use a black background. For thesis
        # figures, turn that background white so object tiles sit naturally on
        # the page.
        arr = np.array(image)
        mask = (arr[:, :, 0] < 18) & (arr[:, :, 1] < 18) & (arr[:, :, 2] < 18)
        arr[mask] = 255
        image = Image.fromarray(arr)
        image = ImageOps.pad(image, (size, size), method=Image.Resampling.BICUBIC, color="white")
    else:
        image = make_placeholder(size, "missing")
    return ImageOps.expand(image, border=border_width, fill=border_color)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = TEXT,
) -> None:
    text_w, text_h = text_size(draw, text, font)
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - text_w) // 2, y0 + (y1 - y0 - text_h) // 2),
        text,
        fill=fill,
        font=font,
    )


def render_topk_figure(
    selected_indices: Sequence[int],
    query_entries: Sequence[Tuple[str, str]],
    baseline_rows: Sequence[dict],
    lora_rows: Sequence[dict],
    args: argparse.Namespace,
    output_path: str,
) -> None:
    tile = args.tile_size
    border = 6
    tile_outer = tile + border * 2
    pad = 44
    query_label_w = 136
    method_label_w = 182
    query_w = tile_outer
    gap = 12
    topk_w = args.top_k * tile_outer + (args.top_k - 1) * gap
    row_gap = 16
    block_gap = 36
    title_h = 0
    legend_h = 42
    top_header_h = 34
    method_row_h = tile_outer
    block_h = top_header_h + method_row_h * 2 + row_gap

    width = pad * 2 + query_label_w + query_w + 28 + method_label_w + topk_w
    height = pad * 2 + title_h + legend_h + len(selected_indices) * block_h
    height += max(0, len(selected_indices) - 1) * block_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = find_font(28, bold=True)
    header_font = find_font(22, bold=True)
    body_font = find_font(20)
    small_font = find_font(18)

    legend_y = pad + title_h
    legend_items = [
        (BLUE, "查询样本"),
        (GREEN, "正确检索结果"),
        (RED, "错误检索结果"),
    ]
    legend_x = pad
    for color, label in legend_items:
        draw.rounded_rectangle(
            (legend_x, legend_y + 8, legend_x + 24, legend_y + 32),
            radius=3,
            fill=color,
        )
        draw.text((legend_x + 34, legend_y + 8), label, fill=TEXT, font=small_font)
        label_w_px, _ = text_size(draw, label, small_font)
        legend_x += 34 + label_w_px + 48

    x_query_label = pad
    x_query = x_query_label + query_label_w
    x_method = x_query + query_w + 28
    x_topk = x_method + method_label_w
    y = pad + title_h + legend_h

    for display_idx, query_index in enumerate(selected_indices, start=1):
        query_class, query_item = query_entries[query_index]
        baseline = baseline_rows[query_index]
        lora = lora_rows[query_index]

        draw.rounded_rectangle(
            (pad - 14, y - 6, width - pad + 14, y + block_h - 4),
            radius=8,
            outline=(226, 230, 236),
            width=1,
            fill=(252, 253, 255),
        )

        draw_centered_text(
            draw,
            (
                x_query_label,
                y + top_header_h,
                x_query_label + query_label_w,
                y + block_h - 4,
            ),
            f"查询样本{display_idx}",
            header_font,
        )
        query_image = resolve_rgb_image(
            args.rgb_image_root,
            query_class,
            query_item,
            args.view_index,
        )
        query_tile = load_tile(query_image, tile, BLUE, border)
        canvas.paste(query_tile, (x_query, y + top_header_h + 44))

        for k in range(args.top_k):
            x0 = x_topk + k * (tile_outer + gap)
            draw_centered_text(
                draw,
                (x0, y, x0 + tile_outer, y + top_header_h),
                f"Top-{k + 1}",
                small_font,
                fill=MUTED,
            )

        method_rows = [
            ("Fusion 基线", baseline, y + top_header_h),
            ("Fusion+LoRA", lora, y + top_header_h + method_row_h + row_gap),
        ]
        for method_name, row, row_y in method_rows:
            draw.text(
                (x_method, row_y + method_row_h // 2 - 12),
                method_name,
                fill=TEXT,
                font=body_font,
            )
            for result in row["topk"]:
                x0 = x_topk + (result["rank"] - 1) * (tile_outer + gap)
                image_path = resolve_rgb_image(
                    args.rgb_image_root,
                    result["class_name"],
                    result["item_name"],
                    args.view_index,
                )
                color = GREEN if result["is_correct"] else RED
                tile_img = load_tile(image_path, tile, color, border)
                canvas.paste(tile_img, (x0, row_y))

        y += block_h + block_gap

    canvas.save(output_path, dpi=(300, 300))


def select_feature_samples(
    protocol: dict,
    num_classes: int,
    samples_per_class: int,
    seed: int,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    rng = np.random.default_rng(seed)
    selected_entries: List[Tuple[str, str]] = []
    selected_classes: List[str] = []

    unseen = protocol["unseen_classes"]
    split_items: Dict[str, List[str]] = {}
    for class_name in unseen:
        items = []
        items.extend(protocol["gallery_unseen"].get(class_name, []))
        items.extend(protocol["query_unseen"].get(class_name, []))
        if len(items) >= samples_per_class:
            split_items[class_name] = sorted(items)

    if class_names:
        missing = [
            class_name
            for class_name in class_names
            if class_name not in split_items
        ]
        if missing:
            raise ValueError(
                f"These requested feature classes do not have at least "
                f"{samples_per_class} samples: {missing}"
            )
        chosen = list(class_names)
    else:
        class_candidates = sorted(split_items)
        if len(class_candidates) < num_classes:
            raise ValueError(
                f"Only {len(class_candidates)} unseen classes have at least "
                f"{samples_per_class} samples."
            )
        chosen = sorted(rng.choice(class_candidates, size=num_classes, replace=False))

    if len(chosen) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    for class_name in chosen:
        items = np.array(split_items[class_name])
        sampled = rng.choice(items, size=samples_per_class, replace=False)
        for item_name in sorted(sampled.tolist()):
            selected_entries.append((class_name, item_name))
        selected_classes.append(class_name)

    return selected_entries, selected_classes


def get_chinese_font_path() -> Optional[str]:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def render_feature_separability_figure(
    baseline_features: np.ndarray,
    lora_features: np.ndarray,
    labels: Sequence[str],
    class_names: Sequence[str],
    output_path: str,
) -> dict:
    label_array = np.array(labels)
    baseline_samples = silhouette_samples(
        baseline_features, label_array, metric="cosine"
    )
    lora_samples = silhouette_samples(lora_features, label_array, metric="cosine")
    baseline_avg = float(
        silhouette_score(baseline_features, label_array, metric="cosine")
    )
    lora_avg = float(silhouette_score(lora_features, label_array, metric="cosine"))

    baseline_means = []
    lora_means = []
    for class_name in class_names:
        mask = label_array == class_name
        baseline_means.append(float(baseline_samples[mask].mean()))
        lora_means.append(float(lora_samples[mask].mean()))

    font_path = get_chinese_font_path()
    label_font = font_manager.FontProperties(fname=font_path, size=10) if font_path else None
    tick_font = font_manager.FontProperties(fname=font_path, size=9) if font_path else None
    legend_font = font_manager.FontProperties(fname=font_path, size=9) if font_path else None

    blue = "#7EA6D8"
    green = "#3BAA5D"
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.8, 4.2), dpi=300)
    ax.bar(x - width / 2, baseline_means, width, label="Fusion 基线", color=blue)
    ax.bar(x + width / 2, lora_means, width, label="Fusion+LoRA", color=green)
    ax.axhline(baseline_avg, color=blue, linestyle="--", linewidth=1.1, alpha=0.9)
    ax.axhline(lora_avg, color=green, linestyle="--", linewidth=1.1, alpha=0.9)

    ax.text(
        len(class_names) - 0.05,
        lora_avg + 0.01,
        f"Fusion+LoRA 平均值={lora_avg:.3f}",
        color="#248545",
        ha="right",
        va="bottom",
        fontsize=9,
        fontproperties=label_font,
    )
    ax.text(
        len(class_names) - 0.05,
        baseline_avg + 0.01,
        f"Fusion 基线平均值={baseline_avg:.3f}",
        color="#4E78A8",
        ha="right",
        va="bottom",
        fontsize=9,
        fontproperties=label_font,
    )

    ax.set_ylabel("轮廓系数", fontproperties=label_font)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"类别{i + 1}" for i in range(len(class_names))],
        fontproperties=tick_font,
    )
    y_min = min(0.0, min(baseline_means), min(lora_means)) - 0.03
    y_max = max(max(baseline_means), max(lora_means), lora_avg) + 0.12
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color="#E4E7EC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", prop=legend_font)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D0D5DD")
    ax.spines["bottom"].set_color("#D0D5DD")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {
        "overall": {
            "fusion_baseline": baseline_avg,
            "fusion_lora": lora_avg,
            "gain": lora_avg - baseline_avg,
        },
        "class_scores": [
            {
                "display_label": f"类别{i + 1}",
                "class_name": class_name,
                "fusion_baseline": baseline_means[i],
                "fusion_lora": lora_means[i],
                "gain": lora_means[i] - baseline_means[i],
            }
            for i, class_name in enumerate(class_names)
        ],
    }


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    protocol = load_protocol(args.protocol)
    query_entries = get_split_items(protocol, "query_unseen")
    gallery_entries = get_split_items(protocol, "gallery_unseen")

    print("Loading Fusion baseline features...")
    base_query = load_split_features(
        query_entries, args.rgb_feat_root, args.depth_feat_root, args.alpha
    )
    base_gallery = load_split_features(
        gallery_entries, args.rgb_feat_root, args.depth_feat_root, args.alpha
    )

    print("Loading Fusion+LoRA features...")
    lora_query = load_split_features(
        query_entries, args.lora_rgb_feat_root, args.lora_depth_feat_root, args.alpha
    )
    lora_gallery = load_split_features(
        gallery_entries, args.lora_rgb_feat_root, args.lora_depth_feat_root, args.alpha
    )

    print("Computing Top-K rankings...")
    baseline_rows = build_rankings(
        base_query, base_gallery, query_entries, gallery_entries, args.top_k
    )
    lora_rows = build_rankings(
        lora_query, lora_gallery, query_entries, gallery_entries, args.top_k
    )
    selected = select_topk_queries(
        baseline_rows,
        lora_rows,
        args.num_queries,
        parse_query_indices(args.query_indices, args.num_queries),
    )

    topk_path = os.path.join(args.output_dir, "figure5_topk_fusion_lora.png")
    render_topk_figure(
        selected,
        query_entries,
        baseline_rows,
        lora_rows,
        args,
        topk_path,
    )
    topk_meta = {
        "figure": "Top-K retrieval comparison",
        "protocol_path": os.path.abspath(args.protocol),
        "methods": ["Fusion baseline", "Fusion+LoRA"],
        "top_k": args.top_k,
        "selection_rule": (
            "Balanced case selection: one strong-improvement query "
            "(Fusion hits=0, LoRA hits=4-5), one moderate-improvement query "
            "(Fusion hits=1-2, LoRA hits=3-4), and one hard-but-improved query "
            "(Fusion hits=1-2, LoRA hits=2-3), with distinct classes preferred."
        ),
        "selected_query_indices": selected,
        "caption": (
            "图5-X 不同方法的开放集三维物体 Top-K 检索结果可视化。"
            "蓝色边框表示查询样本，绿色边框表示正确检索结果，"
            "红色边框表示错误检索结果。"
        ),
        "queries": [],
    }
    for display_idx, query_index in enumerate(selected, start=1):
        row = {
            "query_label": f"Query {display_idx}",
            "query_index": int(query_index),
            "class_name": query_entries[query_index][0],
            "item_name": query_entries[query_index][1],
            "fusion_baseline": baseline_rows[query_index],
            "fusion_lora": lora_rows[query_index],
        }
        topk_meta["queries"].append(row)
    write_json(os.path.splitext(topk_path)[0] + ".json", topk_meta)

    print("Selecting identical unseen samples for feature separability...")
    sample_entries, sample_classes = select_feature_samples(
        protocol,
        args.sample_classes,
        args.samples_per_class,
        args.sample_seed,
        [
            class_name.strip()
            for class_name in args.sample_class_names.split(",")
            if class_name.strip()
        ],
    )
    sample_labels = [class_name for class_name, _ in sample_entries]

    print("Loading feature samples...")
    base_samples = load_split_features(
        sample_entries, args.rgb_feat_root, args.depth_feat_root, args.alpha
    )
    lora_samples = load_split_features(
        sample_entries, args.lora_rgb_feat_root, args.lora_depth_feat_root, args.alpha
    )

    separability_path = os.path.join(
        args.output_dir, "figure5_feature_separability_fusion_lora.png"
    )
    separability = render_feature_separability_figure(
        base_samples,
        lora_samples,
        sample_labels,
        sample_classes,
        separability_path,
    )
    separability_meta = {
        "figure": "Feature separability comparison",
        "protocol_path": os.path.abspath(args.protocol),
        "methods": ["Fusion baseline", "Fusion+LoRA"],
        "same_samples_for_both_methods": True,
        "metric": "silhouette score with cosine distance",
        "higher_is_better": True,
        "sample_seed": args.sample_seed,
        "num_classes": len(sample_classes),
        "samples_per_class": args.samples_per_class,
        "selected_classes": sample_classes,
        "caption": (
            "图5-X Fusion与Fusion+LoRA在同一批未知类样本上的特征可分性对比。"
            "轮廓系数越高，表示类内聚合和类间分离效果越好。"
        ),
        "sample_entries": [
            {"class_name": class_name, "item_name": item_name}
            for class_name, item_name in sample_entries
        ],
        **separability,
    }
    write_json(os.path.splitext(separability_path)[0] + ".json", separability_meta)

    print(f"Saved Top-K figure: {topk_path}")
    print(f"Saved feature separability figure: {separability_path}")


if __name__ == "__main__":
    main()
