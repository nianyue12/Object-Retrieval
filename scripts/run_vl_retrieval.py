import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.clip(norms, 1e-12, None)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)


def entropy_confidence(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] <= 1:
        return np.ones(probs.shape[0], dtype=np.float32)

    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return 1.0 - entropy / np.log(probs.shape[1])


def resolve_text_classes(protocol: dict, text_scope: str):
    if text_scope == "all":
        return list(dict.fromkeys(protocol["seen_classes"] + protocol["unseen_classes"]))
    return list(protocol["unseen_classes"])


def load_split_rgb_depth(
    protocol: dict,
    split_name: str,
    rgb_feat_root: str,
    depth_feat_root: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_feats = []
    depth_feats = []
    labels = []

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
    logits = (feats @ text_prototypes.T) / temperature
    probs = softmax_rows(logits).astype(np.float32)
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
    if mode == "rgb":
        return rgb_branch
    if mode == "depth":
        return depth_branch

    if semantic_fusion == "confidence":
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
    if semantic_similarity == "prob":
        return query_semantic["probs"][start:end] @ gallery_semantic["probs"].T
    if semantic_similarity == "embed":
        return query_semantic["embed"][start:end] @ gallery_semantic["embed"].T

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
    sim_matrix = np.empty(
        (query_visual.shape[0], gallery_visual.shape[0]), dtype=np.float32
    )

    for start in tqdm(
        range(0, query_visual.shape[0], batch_size),
        desc="Computing visual-language similarity",
    ):
        end = min(start + batch_size, query_visual.shape[0])
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
    return f"{value:.2f}".replace(".", "p")


def build_default_save_name(args) -> str:
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
        parts.append(args.metric_style)
    return "_".join(parts) + ".json"


def main():
    args = parse_args()
    if args.prompt_mode in {"coop", "cocoop"} and not args.prompt_ckpt:
        raise ValueError(
            "--prompt_ckpt is required when --prompt_mode coop or cocoop is used."
        )

    protocol = load_protocol(args.protocol)
    text_classes = resolve_text_classes(protocol, args.text_scope)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.prompt_mode == "cocoop":
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
