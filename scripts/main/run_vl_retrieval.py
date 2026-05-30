"""
功能：运行带语义重排的 visual-language 检索。

说明：
    这个脚本会先构建视觉分支和语义分支，
    再按指定策略把二者相似度融合，用于 unseen 检索评估。
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
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
from utils.protocol import get_split_items, load_protocol
from utils.semantic import (
    PROMPT_TEMPLATES,
    build_conditional_semantic_branch,
    build_text_prototypes,
    load_cocoop_prompt_components,
)


def parse_args():
    """解析 visual-language 检索脚本的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run zero-training visual-language retrieval with semantic reranking."
    )
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--mode", choices=["rgb", "depth", "fusion"], default="fusion")
    parser.add_argument("--alpha", type=float, default=ALPHA_FUSION)
    parser.add_argument("--rgb_feat_root", type=str, default=RGB_FEAT_DIR)
    parser.add_argument("--depth_feat_root", type=str, default=DEPTH_FEAT_DIR)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument(
        "--prompt_mode",
        choices=["fixed", "coop", "cocoop"],
        default="fixed",
        help="Text prototype construction mode. Use learned modes with --prompt_ckpt.",
    )
    parser.add_argument(
        "--prompt_ckpt",
        type=str,
        default="",
        help="Checkpoint path produced by train_prompt_coop.py.",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--prompt_batch_size",
        type=int,
        default=32,
        help="Batch size for dynamic prompt inference. Only used by CoCoOp.",
    )
    parser.add_argument(
        "--prompt_chunk_size",
        type=int,
        default=128,
        help="How many flattened prompts to encode at once inside CoCoOp text inference.",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--text_scope",
        choices=["unseen", "all"],
        default="unseen",
    )
    parser.add_argument(
        "--semantic_fusion",
        choices=["fixed", "confidence"],
        default="confidence",
    )
    parser.add_argument(
        "--semantic_similarity",
        choices=["prob", "embed", "hybrid"],
        default="prob",
    )
    parser.add_argument(
        "--combine_strategy",
        choices=["global_blend", "topk_blend", "topk_add"],
        default="global_blend",
    )
    parser.add_argument("--semantic_weight", type=float, default=0.25)
    parser.add_argument("--rerank_topk", type=int, default=300)
    parser.add_argument(
        "--metric_style",
        choices=["hgm2r", "legacy", "both"],
        default="hgm2r",
        help="Primary metric style for reporting. Use both to save legacy and HGM2R metrics together.",
    )
    parser.add_argument("--save_name", type=str, default="")
    return parser.parse_args()


def normalize_rows(feats: np.ndarray) -> np.ndarray:
    """对特征矩阵按行做 L2 归一化。"""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.clip(norms, 1e-12, None)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """对每一行 logits 做 softmax。"""
    # 先减去行最大值，避免 exp 时出现数值溢出。
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)


def entropy_confidence(probs: np.ndarray) -> np.ndarray:
    """根据类别分布熵估计每个样本的语义置信度。"""
    if probs.shape[1] <= 1:
        return np.ones(probs.shape[0], dtype=np.float32)

    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return 1.0 - entropy / np.log(probs.shape[1])


def resolve_text_classes(protocol: dict, text_scope: str):
    """
    功能：决定语义文本库使用哪些类别。

    说明：
        unseen -> 只使用 unseen 类别
        all    -> 同时使用 seen + unseen 类别
    """
    if text_scope == "all":
        # dict.fromkeys 用来去重并保留 seen + unseen 的原始顺序。
        return list(dict.fromkeys(protocol["seen_classes"] + protocol["unseen_classes"]))
    return list(protocol["unseen_classes"])


def load_split_rgb_depth(
    protocol: dict,
    split_name: str,
    rgb_feat_root: str,
    depth_feat_root: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    功能：同时读取某个 split 的 RGB 特征、Depth 特征和标签。
    """
    rgb_feats = []
    depth_feats = []
    labels = []

    # RGB 和 Depth 同时加载，后续可以按不同 mode 复用同一份数据。
    for cls, item in get_split_items(protocol, split_name):
        rgb_path = os.path.join(rgb_feat_root, cls, item)
        depth_path = os.path.join(depth_feat_root, cls, item)

        rgb_feats.append(load_feature(rgb_path, aggregation="mean"))
        depth_feats.append(load_feature(depth_path, aggregation="mean"))
        labels.append(cls)

    return (
        normalize_rows(np.stack(rgb_feats).astype(np.float32)),
        normalize_rows(np.stack(depth_feats).astype(np.float32)),
        np.array(labels),
    )


def build_visual_features(
    rgb_feats: np.ndarray,
    depth_feats: np.ndarray,
    mode: str,
    alpha: float,
) -> np.ndarray:
    """根据 mode 构造最终视觉分支特征。"""
    if mode == "rgb":
        return rgb_feats
    if mode == "depth":
        return depth_feats
    return normalize_rows(alpha * rgb_feats + (1.0 - alpha) * depth_feats)


def build_semantic_branch(
    feats: np.ndarray,
    text_prototypes: np.ndarray,
    temperature: float,
) -> Dict[str, np.ndarray]:
    """
    功能：基于固定文本原型构建单个模态的语义分支输出。
    """
    # logits 表示视觉特征对各个文本类别原型的匹配程度。
    logits = (feats @ text_prototypes.T) / temperature
    probs = softmax_rows(logits).astype(np.float32)
    # 用类别概率加权文本原型，得到一个连续语义嵌入。
    semantic_embed = normalize_rows(probs @ text_prototypes).astype(np.float32)
    confidence = entropy_confidence(probs).astype(np.float32)

    return {
        "logits": logits.astype(np.float32),
        "probs": normalize_rows(probs),
        "embed": semantic_embed,
        "confidence": confidence,
    }


def fuse_semantic_branches(
    rgb_branch: Dict[str, np.ndarray],
    depth_branch: Dict[str, np.ndarray],
    mode: str,
    alpha: float,
    semantic_fusion: str,
    text_prototypes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    功能：融合 RGB / Depth 两个模态的语义分支。

    说明：
        confidence 模式会根据每个模态自己的语义置信度动态分配权重。
    """
    if mode == "rgb":
        return rgb_branch
    if mode == "depth":
        return depth_branch

    if semantic_fusion == "confidence":
        # 置信度越高的模态，在语义融合时权重越大。
        rgb_weight = alpha * rgb_branch["confidence"]
        depth_weight = (1.0 - alpha) * depth_branch["confidence"]
        denom = np.clip(rgb_weight + depth_weight, 1e-12, None)
        rgb_weight = rgb_weight / denom
        depth_weight = depth_weight / denom
    else:
        rgb_weight = np.full(rgb_branch["logits"].shape[0], alpha, dtype=np.float32)
        depth_weight = np.full(
            depth_branch["logits"].shape[0], 1.0 - alpha, dtype=np.float32
        )

    fused_logits = (
        rgb_weight[:, None] * rgb_branch["logits"]
        + depth_weight[:, None] * depth_branch["logits"]
    ).astype(np.float32)
    fused_probs = softmax_rows(fused_logits).astype(np.float32)
    fused_embed = normalize_rows(fused_probs @ text_prototypes).astype(np.float32)

    # 记录平均权重，方便结果 JSON 里诊断 RGB/Depth 语义分支贡献。
    return {
        "logits": fused_logits,
        "probs": normalize_rows(fused_probs),
        "embed": fused_embed,
        "confidence": np.maximum(rgb_branch["confidence"], depth_branch["confidence"]),
        "rgb_weight_mean": np.array([float(rgb_weight.mean())], dtype=np.float32),
        "depth_weight_mean": np.array([float(depth_weight.mean())], dtype=np.float32),
    }


def compute_text_top1_acc(
    probs: np.ndarray,
    class_names,
    labels: np.ndarray,
) -> float:
    """根据语义分支概率计算文本 top-1 诊断准确率。"""
    pred_indices = probs.argmax(axis=1)
    pred_labels = np.array(class_names)[pred_indices]
    return float(np.mean(pred_labels == labels))


def build_semantic_similarity(
    query_semantic: Dict[str, np.ndarray],
    gallery_semantic: Dict[str, np.ndarray],
    semantic_similarity: str,
    start: int,
    end: int,
) -> np.ndarray:
    """
    功能：构造 query 与 gallery 的语义相似度。

    支持：
        prob   -> 概率分布相似度
        embed  -> 语义嵌入相似度
        hybrid -> 二者平均
    """
    if semantic_similarity == "prob":
        return query_semantic["probs"][start:end] @ gallery_semantic["probs"].T
    if semantic_similarity == "embed":
        return query_semantic["embed"][start:end] @ gallery_semantic["embed"].T

    # hybrid 同时利用类别概率分布和加权文本嵌入。
    prob_sim = query_semantic["probs"][start:end] @ gallery_semantic["probs"].T
    embed_sim = query_semantic["embed"][start:end] @ gallery_semantic["embed"].T
    return 0.5 * (prob_sim + embed_sim)


def blend_similarity(
    visual_sim: np.ndarray,
    semantic_sim: np.ndarray,
    semantic_weight: float,
    combine_strategy: str,
    rerank_topk: int,
) -> np.ndarray:
    """
    功能：把视觉相似度和语义相似度按指定策略融合。
    """
    if semantic_weight <= 0.0:
        return visual_sim.astype(np.float32)

    if combine_strategy == "global_blend":
        return (
            (1.0 - semantic_weight) * visual_sim + semantic_weight * semantic_sim
        ).astype(np.float32)

    if rerank_topk <= 0:
        return visual_sim.astype(np.float32)

    rerank_topk = min(rerank_topk, visual_sim.shape[1])
    blended = visual_sim.copy()
    # top-k 策略只调整视觉排名靠前的候选，避免语义分支影响整个 gallery。
    topk_indices = np.argpartition(-visual_sim, rerank_topk - 1, axis=1)[
        :, :rerank_topk
    ]
    row_indices = np.arange(visual_sim.shape[0])[:, None]

    if combine_strategy == "topk_add":
        blended[row_indices, topk_indices] = (
            visual_sim[row_indices, topk_indices]
            + semantic_weight * semantic_sim[row_indices, topk_indices]
        )
    else:
        blended[row_indices, topk_indices] = (
            (1.0 - semantic_weight) * visual_sim[row_indices, topk_indices]
            + semantic_weight * semantic_sim[row_indices, topk_indices]
        )

    return blended.astype(np.float32)


def compute_combined_similarity(
    query_visual: np.ndarray,
    gallery_visual: np.ndarray,
    query_semantic: Dict[str, np.ndarray],
    gallery_semantic: Dict[str, np.ndarray],
    batch_size: int,
    semantic_similarity: str,
    semantic_weight: float,
    combine_strategy: str,
    rerank_topk: int,
) -> np.ndarray:
    """
    功能：分 batch 计算最终 visual-language 相似度矩阵。
    """
    sim_matrix = np.empty(
        (query_visual.shape[0], gallery_visual.shape[0]), dtype=np.float32
    )

    for start in tqdm(
        range(0, query_visual.shape[0], batch_size),
        desc="Computing visual-language similarity",
    ):
        end = min(start + batch_size, query_visual.shape[0])
        # 视觉相似度和语义相似度都按 query batch 分块计算。
        visual_sim = query_visual[start:end] @ gallery_visual.T
        semantic_sim = build_semantic_similarity(
            query_semantic,
            gallery_semantic,
            semantic_similarity,
            start,
            end,
        )
        sim_matrix[start:end] = blend_similarity(
            visual_sim=visual_sim,
            semantic_sim=semantic_sim,
            semantic_weight=semantic_weight,
            combine_strategy=combine_strategy,
            rerank_topk=rerank_topk,
        )

    return sim_matrix


def float_tag(value: float) -> str:
    """把浮点数转成适合文件名的字符串。"""
    return f"{value:.2f}".replace(".", "p")


def build_default_save_name(args) -> str:
    """根据当前配置生成默认结果文件名。"""
    parts = [
        "vl",
        args.mode,
    ]
    if args.prompt_mode != "fixed":
        parts.append(args.prompt_mode)
    parts.extend([
        args.text_scope,
        args.semantic_fusion,
        args.semantic_similarity,
        args.combine_strategy,
        f"sw{float_tag(args.semantic_weight)}",
    ])
    if args.mode == "fusion":
        parts.append(f"a{float_tag(args.alpha)}")
    if args.combine_strategy != "global_blend":
        parts.append(f"topk{args.rerank_topk}")
    if args.metric_style != "legacy":
        # legacy 是历史默认命名，hgm2r/both 才显式写进文件名。
        parts.append(args.metric_style)
    return "_".join(parts) + ".json"


def main():
    """脚本入口：构建视觉/语义分支，评估检索并保存结果。"""
    args = parse_args()
    if args.prompt_mode in {"coop", "cocoop"} and not args.prompt_ckpt:
        raise ValueError(
            "--prompt_ckpt is required when --prompt_mode coop or cocoop is used."
        )

    protocol = load_protocol(args.protocol)
    text_classes = resolve_text_classes(protocol, args.text_scope)

    # 先准备 query / gallery 的双模态特征
    gallery_rgb, gallery_depth, gallery_labels = load_split_rgb_depth(
        protocol,
        "gallery_unseen",
        args.rgb_feat_root,
        args.depth_feat_root,
    )
    query_rgb, query_depth, query_labels = load_split_rgb_depth(
        protocol,
        "query_unseen",
        args.rgb_feat_root,
        args.depth_feat_root,
    )

    gallery_visual = build_visual_features(
        gallery_rgb,
        gallery_depth,
        args.mode,
        args.alpha,
    )
    query_visual = build_visual_features(
        query_rgb,
        query_depth,
        args.mode,
        args.alpha,
    )

    # 语义分支分两种路线：CoCoOp 动态生成文本特征，其余模式先生成固定文本原型。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.prompt_mode == "cocoop":
        # CoCoOp 会根据每个样本特征动态生成文本分支
        model, prompt_learner, text_encoder, _ = load_cocoop_prompt_components(
            text_classes,
            clip_model=args.clip_model,
            device=device,
            prompt_checkpoint=args.prompt_ckpt,
        )
        gallery_semantic = build_conditional_semantic_branch(
            gallery_visual,
            model,
            prompt_learner,
            text_encoder,
            temperature=args.temperature,
            batch_size=args.prompt_batch_size,
            prompt_chunk_size=args.prompt_chunk_size,
            desc="Building gallery semantic branch",
        )
        query_semantic = build_conditional_semantic_branch(
            query_visual,
            model,
            prompt_learner,
            text_encoder,
            temperature=args.temperature,
            batch_size=args.prompt_batch_size,
            prompt_chunk_size=args.prompt_chunk_size,
            desc="Building query semantic branch",
        )
    else:
        # fixed / coop 模式先构建固定文本原型，再分别生成 RGB / Depth 语义分支
        text_prototypes = build_text_prototypes(
            text_classes,
            clip_model=args.clip_model,
            device=device,
            templates=PROMPT_TEMPLATES,
            prompt_checkpoint=args.prompt_ckpt or None,
        ).astype(np.float32)

        gallery_rgb_sem = build_semantic_branch(
            gallery_rgb,
            text_prototypes,
            args.temperature,
        )
        gallery_depth_sem = build_semantic_branch(
            gallery_depth,
            text_prototypes,
            args.temperature,
        )
        query_rgb_sem = build_semantic_branch(
            query_rgb,
            text_prototypes,
            args.temperature,
        )
        query_depth_sem = build_semantic_branch(
            query_depth,
            text_prototypes,
            args.temperature,
        )

        gallery_semantic = fuse_semantic_branches(
            gallery_rgb_sem,
            gallery_depth_sem,
            args.mode,
            args.alpha,
            args.semantic_fusion,
            text_prototypes,
        )
        query_semantic = fuse_semantic_branches(
            query_rgb_sem,
            query_depth_sem,
            args.mode,
            args.alpha,
            args.semantic_fusion,
            text_prototypes,
        )

    print(f"Seen classes used for training protocol: {protocol['seen_classes']}")
    print(f"Unseen classes used for retrieval: {protocol['unseen_classes']}")
    print(f"Text bank scope: {args.text_scope} ({len(text_classes)} classes)")
    print(f"Prompt mode: {args.prompt_mode}")
    if args.prompt_mode == "cocoop":
        print(f"Prompt batch size: {args.prompt_batch_size}")
        print(f"Prompt chunk size: {args.prompt_chunk_size}")
    print(f"Gallery size: {gallery_visual.shape[0]}")
    print(f"Query size: {query_visual.shape[0]}")

    # 融合视觉与语义相似度，得到最终检索矩阵
    sim_matrix = compute_combined_similarity(
        query_visual=query_visual,
        gallery_visual=gallery_visual,
        query_semantic=query_semantic,
        gallery_semantic=gallery_semantic,
        batch_size=args.batch_size,
        semantic_similarity=args.semantic_similarity,
        semantic_weight=args.semantic_weight,
        combine_strategy=args.combine_strategy,
        rerank_topk=args.rerank_topk,
    )

    eval_result = evaluate_retrieval(
        sim_matrix,
        gallery_labels,
        query_labels,
        metric_style=args.metric_style,
    )
    primary_style = eval_result["primary_style"]
    primary_metrics = eval_result["metrics"]
    metrics_by_style = eval_result["metrics_by_style"]

    query_text_top1_acc = compute_text_top1_acc(
        query_semantic["probs"], text_classes, query_labels
    )
    gallery_text_top1_acc = compute_text_top1_acc(
        gallery_semantic["probs"], text_classes, gallery_labels
    )

    # 保存结果时同时写入语义诊断指标，便于分析 rerank 是否真的有帮助。
    os.makedirs(UNSEEN_RESULT_DIR, exist_ok=True)
    save_name = args.save_name or build_default_save_name(args)
    save_path = os.path.join(UNSEEN_RESULT_DIR, save_name)

    output = {
        "method": "visual_language_rerank",
        "mode": args.mode,
        "protocol_path": args.protocol,
        "clip_model": args.clip_model,
        "prompt_mode": args.prompt_mode,
        "prompt_ckpt": args.prompt_ckpt,
        "seen_classes": protocol["seen_classes"],
        "unseen_classes": protocol["unseen_classes"],
        "text_classes": text_classes,
        "gallery_size": int(gallery_visual.shape[0]),
        "query_size": int(query_visual.shape[0]),
        "alpha_fusion": float(args.alpha),
        "temperature": float(args.temperature),
        "prompt_batch_size": int(args.prompt_batch_size),
        "prompt_chunk_size": int(args.prompt_chunk_size),
        "text_scope": args.text_scope,
        "semantic_fusion": args.semantic_fusion,
        "semantic_similarity": args.semantic_similarity,
        "combine_strategy": args.combine_strategy,
        "semantic_weight": float(args.semantic_weight),
        "rerank_topk": int(args.rerank_topk),
        "metric_style": args.metric_style,
        "primary_metric_style": primary_style,
        "query_text_top1_acc": query_text_top1_acc,
        "gallery_text_top1_acc": gallery_text_top1_acc,
        "metrics": primary_metrics,
        "metrics_by_style": metrics_by_style,
    }

    if "rgb_weight_mean" in query_semantic:
        output["query_rgb_semantic_weight_mean"] = float(
            query_semantic["rgb_weight_mean"][0]
        )
        output["query_depth_semantic_weight_mean"] = float(
            query_semantic["depth_weight_mean"][0]
        )
        output["gallery_rgb_semantic_weight_mean"] = float(
            gallery_semantic["rgb_weight_mean"][0]
        )
        output["gallery_depth_semantic_weight_mean"] = float(
            gallery_semantic["depth_weight_mean"][0]
        )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Saved: {save_path}")
    if args.metric_style == "both":
        print(
            "visual-language unseen retrieval [hgm2r]: "
            f"{format_metric_report(metrics_by_style['hgm2r'])}"
        )
        print(
            "visual-language unseen retrieval [legacy]: "
            f"{format_metric_report(metrics_by_style['legacy'])}"
        )
    else:
        print(
            f"visual-language unseen retrieval [{primary_style}]: "
            f"{format_metric_report(primary_metrics)}"
        )
    print(
        "semantic top1 diagnostics: "
        f"query={query_text_top1_acc:.4f}, gallery={gallery_text_top1_acc:.4f}"
    )


if __name__ == "__main__":
    main()
