"""
功能：在 unseen 类别上运行基础零样本检索。

说明：
    这里的相似度只来自视觉特征，
    支持 RGB、Depth 和二者加权融合三种模式。
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import (
    ALPHA_FUSION,
    BATCH_SIZE,
    DEFAULT_PROTOCOL_PATH,
    DEPTH_FEAT_DIR,
    RGB_FEAT_DIR,
    UNSEEN_RESULT_DIR,
)
from utils.features import load_feature
from utils.metrics import evaluate_retrieval, format_metric_report
from utils.protocol import get_split_items, load_protocol, materialize_split_paths


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot retrieval on unseen classes."
    )
    parser.add_argument("--mode", choices=["rgb", "depth", "fusion"], required=True)
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--alpha", type=float, default=ALPHA_FUSION)
    parser.add_argument("--rgb_feat_root", type=str, default=RGB_FEAT_DIR)
    parser.add_argument("--depth_feat_root", type=str, default=DEPTH_FEAT_DIR)
    parser.add_argument(
        "--metric_style",
        choices=["hgm2r", "legacy", "both"],
        default="hgm2r",
        help="Primary metric style for reporting. Use both to save legacy and HGM2R metrics together.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="",
        help="Optional custom output file name under results/unseen_retrieval.",
    )
    return parser.parse_args()


def load_single_modality(protocol, split_name, feat_root):
    """
    功能：读取某个 split 的单模态特征和标签。
    """
    entries = materialize_split_paths(protocol, split_name, feat_root)
    feats, labels = [], []
    for cls, path in entries:
        # 单模态路径已经由 materialize_split_paths 拼好，直接读取特征。
        feats.append(load_feature(path, aggregation="mean"))
        labels.append(cls)
    return np.stack(feats), np.array(labels)


def load_fusion(protocol, split_name, alpha, rgb_feat_root, depth_feat_root):
    """
    功能：读取 RGB / Depth 特征并做加权融合。
    """
    feats, labels = [], []
    for cls, item in get_split_items(protocol, split_name):
        # fusion 模式要求 RGB 和 Depth 使用相同的 class/item 索引。
        rgb_feat = load_feature(
            os.path.join(rgb_feat_root, cls, item),
            aggregation="mean",
        )
        depth_feat = load_feature(
            os.path.join(depth_feat_root, cls, item),
            aggregation="mean",
        )
        fused = alpha * rgb_feat + (1.0 - alpha) * depth_feat
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        feats.append(fused)
        labels.append(cls)
    return np.stack(feats), np.array(labels)


def normalize_rows(feats):
    """对特征矩阵按行做 L2 归一化。"""
    # 后续矩阵乘法直接作为余弦相似度。
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def compute_similarity(query_feats, gallery_feats):
    """
    功能：分 batch 计算 query 与 gallery 的余弦相似度矩阵。
    """
    sim = np.zeros((query_feats.shape[0], gallery_feats.shape[0]), dtype=np.float32)
    for start in tqdm(range(0, query_feats.shape[0], BATCH_SIZE), desc="Computing similarity"):
        end = min(start + BATCH_SIZE, query_feats.shape[0])
        # 分块计算可以避免一次性构造过大的中间矩阵。
        sim[start:end] = query_feats[start:end] @ gallery_feats.T
    return sim


def decorate_save_name(base_name, metric_style):
    """根据指标风格给结果文件名加后缀。"""
    if metric_style == "legacy":
        return base_name
    stem, ext = os.path.splitext(base_name)
    return f"{stem}_{metric_style}{ext}"


def main():
    """脚本入口：加载特征、计算相似度、评估并保存结果。"""
    args = parse_args()
    protocol = load_protocol(args.protocol)

    # 按 mode 选择读取 RGB、Depth 或融合特征，后续评估流程保持一致。
    if args.mode == "rgb":
        gallery_feats, gallery_labels = load_single_modality(
            protocol, "gallery_unseen", args.rgb_feat_root
        )
        query_feats, query_labels = load_single_modality(
            protocol, "query_unseen", args.rgb_feat_root
        )
    elif args.mode == "depth":
        gallery_feats, gallery_labels = load_single_modality(
            protocol, "gallery_unseen", args.depth_feat_root
        )
        query_feats, query_labels = load_single_modality(
            protocol, "query_unseen", args.depth_feat_root
        )
    else:
        gallery_feats, gallery_labels = load_fusion(
            protocol,
            "gallery_unseen",
            args.alpha,
            args.rgb_feat_root,
            args.depth_feat_root,
        )
        query_feats, query_labels = load_fusion(
            protocol,
            "query_unseen",
            args.alpha,
            args.rgb_feat_root,
            args.depth_feat_root,
        )

    # 归一化后做点积，相当于余弦相似度
    gallery_feats = normalize_rows(gallery_feats)
    query_feats = normalize_rows(query_feats)

    print(f"Seen classes used for training protocol: {protocol['seen_classes']}")
    print(f"Unseen classes used for retrieval: {protocol['unseen_classes']}")
    print(f"Gallery size: {gallery_feats.shape[0]}")
    print(f"Query size: {query_feats.shape[0]}")

    # 计算检索相似度并评估指标
    sim_matrix = compute_similarity(query_feats, gallery_feats)

    eval_result = evaluate_retrieval(
        sim_matrix,
        gallery_labels,
        query_labels,
        metric_style=args.metric_style,
    )
    primary_style = eval_result["primary_style"]
    primary_metrics = eval_result["metrics"]
    metrics_by_style = eval_result["metrics_by_style"]

    os.makedirs(UNSEEN_RESULT_DIR, exist_ok=True)
    if args.mode == "fusion":
        alpha_tag = f"{args.alpha:.2f}".replace(".", "p")
        base_save_name = f"{args.mode}_zero_shot_alpha{alpha_tag}.json"
    else:
        base_save_name = f"{args.mode}_zero_shot.json"
    save_name = args.save_name or decorate_save_name(base_save_name, args.metric_style)
    save_path = os.path.join(UNSEEN_RESULT_DIR, save_name)

    # 保存完整实验配置，便于结果文件独立复现。
    output = {
        "mode": args.mode,
        "protocol_path": args.protocol,
        "seen_classes": protocol["seen_classes"],
        "unseen_classes": protocol["unseen_classes"],
        "gallery_size": int(gallery_feats.shape[0]),
        "query_size": int(query_feats.shape[0]),
        "alpha_fusion": float(args.alpha),
        "metric_style": args.metric_style,
        "primary_metric_style": primary_style,
        "metrics": primary_metrics,
        "metrics_by_style": metrics_by_style,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Saved: {save_path}")
    if args.metric_style == "both":
        print(
            f"{args.mode} zero-shot unseen retrieval [hgm2r]: "
            f"{format_metric_report(metrics_by_style['hgm2r'])}"
        )
        print(
            f"{args.mode} zero-shot unseen retrieval [legacy]: "
            f"{format_metric_report(metrics_by_style['legacy'])}"
        )
    else:
        print(
            f"{args.mode} zero-shot unseen retrieval [{primary_style}]: "
            f"{format_metric_report(primary_metrics)}"
        )


if __name__ == "__main__":
    main()
