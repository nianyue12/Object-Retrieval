"""
Evaluate safe Fusion-baseline PEFT retrieval.

This script is intentionally separate from the older VL retrieval script. It
never builds unseen-class text prototypes for ranking. Seen-anchor methods use
only protocol["seen_classes"] as text anchors; unseen labels are used only by
the final metric evaluator.
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import (  # noqa: E402
    ALPHA_FUSION,
    BATCH_SIZE,
    DEFAULT_PROTOCOL_PATH,
    DEPTH_FEAT_DIR,
    RGB_FEAT_DIR,
    UNSEEN_RESULT_DIR,
)
from utils.features import load_feature  # noqa: E402
from utils.metrics import evaluate_retrieval, format_metric_report  # noqa: E402
from utils.protocol import get_split_items, load_protocol  # noqa: E402


SAFE_METHODS = {
    "fusion",
    "adapter",
    "lora",
    "coop_seen_anchor",
    "cocoop_seen_anchor",
}
SEEN_ANCHOR_METHODS = {"coop_seen_anchor", "cocoop_seen_anchor"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run safe Fusion-baseline PEFT retrieval on unseen query/gallery."
    )
    parser.add_argument("--method", choices=sorted(SAFE_METHODS), default="fusion")
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--alpha", type=float, default=ALPHA_FUSION)
    parser.add_argument("--rgb_feat_root", type=str, default=RGB_FEAT_DIR)
    parser.add_argument("--depth_feat_root", type=str, default=DEPTH_FEAT_DIR)
    parser.add_argument(
        "--lora_rgb_feat_root",
        type=str,
        default="",
        help="RGB feature root extracted with the LoRA-adapted CLIP image encoder.",
    )
    parser.add_argument(
        "--lora_depth_feat_root",
        type=str,
        default="",
        help="Depth feature root extracted with the LoRA-adapted CLIP image encoder.",
    )
    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default="",
        help="Checkpoint from scripts/adapter/train_fusion_adapter.py.",
    )
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--prompt_ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--seen_anchor_weight",
        type=float,
        default=0.25,
        help="Lambda for seen-anchor response similarity.",
    )
    parser.add_argument(
        "--seen_anchor_similarity",
        choices=["cosine", "dot", "bhattacharyya", "centered_cosine"],
        default="cosine",
        help=(
            "Similarity used between seen-anchor response distributions. "
            "cosine matches the original implementation; bhattacharyya is often "
            "more stable for probability distributions."
        ),
    )
    parser.add_argument(
        "--seen_anchor_power",
        type=float,
        default=1.0,
        help=(
            "Optional power transform for seen-anchor probabilities before "
            "similarity. Values below 1.0 soften over-confident responses."
        ),
    )
    parser.add_argument(
        "--text_scope",
        choices=["seen"],
        default="seen",
        help="Seen-anchor methods are only allowed to use seen text anchors.",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--prompt_batch_size", type=int, default=32)
    parser.add_argument("--prompt_chunk_size", type=int, default=128)
    parser.add_argument(
        "--metric_style",
        choices=["hgm2r", "legacy", "both"],
        default="hgm2r",
    )
    parser.add_argument("--output_dir", type=str, default=UNSEEN_RESULT_DIR)
    parser.add_argument("--save_name", type=str, default="")
    return parser.parse_args()


def resolve_device(device_arg: str):
    import torch

    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_rows(feats: np.ndarray) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.clip(norms, 1e-12, None)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)


def float_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


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


def build_fusion_features(
    rgb_feats: np.ndarray,
    depth_feats: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return normalize_rows(alpha * rgb_feats + (1.0 - alpha) * depth_feats)


def load_fusion_split(
    protocol: dict,
    split_name: str,
    rgb_feat_root: str,
    depth_feat_root: str,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb_feats, depth_feats, labels = load_split_rgb_depth(
        protocol,
        split_name,
        rgb_feat_root,
        depth_feat_root,
    )
    return build_fusion_features(rgb_feats, depth_feats, alpha), labels


def load_adapter(checkpoint_path: str, device) -> Tuple[object, dict]:
    from models.clip_adapter import CLIPResidualAdapter

    if not checkpoint_path:
        raise ValueError("--adapter_ckpt is required for --method adapter.")

    checkpoint = safe_torch_load(checkpoint_path, device=device)
    adapter = CLIPResidualAdapter(
        dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
        residual_scale=float(checkpoint.get("residual_scale", 0.2)),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    return adapter, checkpoint


def apply_adapter_to_features(
    features: np.ndarray,
    adapter,
    batch_size: int,
) -> np.ndarray:
    import torch

    device = next(adapter.parameters()).device
    adapted = np.empty_like(features, dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(
            range(0, features.shape[0], batch_size),
            desc="Applying post-fusion Adapter",
        ):
            end = min(start + batch_size, features.shape[0])
            batch = torch.from_numpy(features[start:end]).to(device)
            adapted[start:end] = adapter(batch).cpu().numpy().astype(np.float32)

    return normalize_rows(adapted)


def compute_similarity(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    batch_size: int,
    desc: str = "Computing cosine similarity",
) -> np.ndarray:
    sim = np.empty((query_feats.shape[0], gallery_feats.shape[0]), dtype=np.float32)
    for start in tqdm(range(0, query_feats.shape[0], batch_size), desc=desc):
        end = min(start + batch_size, query_feats.shape[0])
        sim[start:end] = query_feats[start:end] @ gallery_feats.T
    return sim


def compute_seen_anchor_similarity(
    query_visual: np.ndarray,
    gallery_visual: np.ndarray,
    query_response: np.ndarray,
    gallery_response: np.ndarray,
    seen_anchor_weight: float,
    seen_anchor_similarity: str,
    seen_anchor_power: float,
    batch_size: int,
) -> np.ndarray:
    if not 0.0 <= seen_anchor_weight <= 1.0:
        raise ValueError("--seen_anchor_weight must be in [0, 1].")

    query_response, gallery_response = prepare_seen_anchor_responses(
        query_response,
        gallery_response,
        similarity=seen_anchor_similarity,
        power=seen_anchor_power,
    )

    sim = np.empty((query_visual.shape[0], gallery_visual.shape[0]), dtype=np.float32)
    for start in tqdm(
        range(0, query_visual.shape[0], batch_size),
        desc="Computing Fusion + seen-anchor similarity",
    ):
        end = min(start + batch_size, query_visual.shape[0])
        visual_sim = query_visual[start:end] @ gallery_visual.T
        anchor_sim = query_response[start:end] @ gallery_response.T
        sim[start:end] = (
            (1.0 - seen_anchor_weight) * visual_sim
            + seen_anchor_weight * anchor_sim
        ).astype(np.float32)
    return sim


def renormalize_prob_rows(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float32), 1e-12, None)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def prepare_seen_anchor_responses(
    query_response: np.ndarray,
    gallery_response: np.ndarray,
    similarity: str,
    power: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if power <= 0.0:
        raise ValueError("--seen_anchor_power must be positive.")

    query_response = renormalize_prob_rows(query_response)
    gallery_response = renormalize_prob_rows(gallery_response)

    if power != 1.0:
        query_response = renormalize_prob_rows(np.power(query_response, power))
        gallery_response = renormalize_prob_rows(np.power(gallery_response, power))

    if similarity == "dot":
        return query_response.astype(np.float32), gallery_response.astype(np.float32)

    if similarity == "bhattacharyya":
        # sqrt(prob) vectors have unit L2 norm, and their dot product is the
        # Bhattacharyya coefficient. It is less brittle than cosine for very
        # peaked probability distributions.
        return (
            np.sqrt(query_response).astype(np.float32),
            np.sqrt(gallery_response).astype(np.float32),
        )

    if similarity == "centered_cosine":
        center = np.concatenate([query_response, gallery_response], axis=0).mean(
            axis=0,
            keepdims=True,
        )
        return (
            normalize_rows(query_response - center).astype(np.float32),
            normalize_rows(gallery_response - center).astype(np.float32),
        )

    if similarity == "cosine":
        return (
            normalize_rows(query_response).astype(np.float32),
            normalize_rows(gallery_response).astype(np.float32),
        )

    raise ValueError(f"Unsupported seen_anchor_similarity: {similarity}")


def validate_seen_anchor_classes(protocol: dict, text_classes, text_scope: str) -> None:
    if text_scope != "seen":
        raise ValueError("Seen-anchor methods must use text_scope='seen'.")

    seen_set = set(protocol["seen_classes"])
    unseen_set = set(protocol["unseen_classes"])
    text_set = set(text_classes)

    if text_set != seen_set:
        raise ValueError("Seen-anchor text classes must exactly match protocol seen_classes.")
    if text_set & unseen_set:
        raise ValueError("Seen-anchor text classes must not contain unseen classes.")


def build_coop_seen_response(
    features: np.ndarray,
    seen_classes,
    clip_model: str,
    prompt_ckpt: str,
    temperature: float,
    device,
) -> np.ndarray:
    from utils.semantic import PROMPT_TEMPLATES, build_text_prototypes

    if not prompt_ckpt:
        raise ValueError("--prompt_ckpt is required for --method coop_seen_anchor.")

    text_prototypes = build_text_prototypes(
        seen_classes,
        clip_model=clip_model,
        device=device,
        templates=PROMPT_TEMPLATES,
        prompt_checkpoint=prompt_ckpt,
    ).astype(np.float32)
    logits = (features @ text_prototypes.T) / max(float(temperature), 1e-6)
    return softmax_rows(logits).astype(np.float32)


def build_cocoop_seen_response(
    features: np.ndarray,
    seen_classes,
    clip_model: str,
    prompt_ckpt: str,
    temperature: float,
    prompt_batch_size: int,
    prompt_chunk_size: int,
    device,
    desc: str,
) -> np.ndarray:
    from utils.semantic import (
        build_conditional_semantic_branch,
        load_cocoop_prompt_components,
    )

    if not prompt_ckpt:
        raise ValueError("--prompt_ckpt is required for --method cocoop_seen_anchor.")

    _, prompt_learner, text_encoder, _ = load_cocoop_prompt_components(
        seen_classes,
        clip_model=clip_model,
        device=device,
        prompt_checkpoint=prompt_ckpt,
    )
    branch = build_conditional_semantic_branch(
        features,
        model=None,
        prompt_learner=prompt_learner,
        text_encoder=text_encoder,
        temperature=temperature,
        batch_size=prompt_batch_size,
        prompt_chunk_size=prompt_chunk_size,
        desc=desc,
    )
    return softmax_rows(branch["logits"]).astype(np.float32)


def require_lora_feature_roots(args) -> Tuple[str, str]:
    if not args.lora_rgb_feat_root or not args.lora_depth_feat_root:
        raise ValueError(
            "--lora_rgb_feat_root and --lora_depth_feat_root are required for "
            "--method lora. Extract features with scripts/lora/extract_clip_features_lora.py first."
        )
    return args.lora_rgb_feat_root, args.lora_depth_feat_root


def default_save_name(args) -> str:
    suffix = args.metric_style
    if args.method == "fusion":
        return f"fusion_baseline_{suffix}.json"
    if args.method == "adapter":
        return f"fusion_adapter_visual_only_{suffix}.json"
    if args.method == "lora":
        return f"fusion_lora_visual_only_{suffix}.json"
    if args.method == "coop_seen_anchor":
        return (
            f"fusion_coop_seen_anchor_w{float_tag(args.seen_anchor_weight)}"
            f"_{args.seen_anchor_similarity}_p{float_tag(args.seen_anchor_power)}"
            f"_{suffix}.json"
        )
    if args.method == "cocoop_seen_anchor":
        return (
            f"fusion_cocoop_seen_anchor_w{float_tag(args.seen_anchor_weight)}"
            f"_{args.seen_anchor_similarity}_p{float_tag(args.seen_anchor_power)}"
            f"_{suffix}.json"
        )
    raise ValueError(f"Unsupported method: {args.method}")


def save_json_with_fallback(data: dict, save_path: str) -> str:
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return save_path
    except PermissionError:
        parent_dir = os.path.dirname(os.path.dirname(save_path))
        fallback_path = os.path.join(parent_dir, os.path.basename(save_path))
        if fallback_path == save_path:
            raise
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(
            f"Warning: could not write to {save_path}; "
            f"saved fallback result to {fallback_path}"
        )
        return fallback_path


def method_metadata(args, adapter_checkpoint: Optional[dict]) -> Dict[str, object]:
    common = {
        "base_feature": "rgb_depth_fusion",
        "alpha_fusion": float(args.alpha),
        "train_split": "train_seen",
        "val_split": "val_seen",
        "query_split": "query_unseen",
        "gallery_split": "gallery_unseen",
        "text_anchor_scope": "none",
        "uses_seen_class_names_as_anchors": False,
        "uses_unseen_class_names_for_ranking": False,
        "uses_unseen_labels_for_ranking": False,
        "uses_gallery_labels_for_ranking": False,
        "unseen_labels_used_only_for_metrics": True,
    }

    if args.method == "fusion":
        common.update(
            {
                "feature_source": "cached_base_clip_features",
                "ranking_score_type": "cosine_similarity_of_rgb_depth_fusion_features",
            }
        )
    elif args.method == "adapter":
        common.update(
            {
                "feature_source": "cached_base_clip_features_plus_post_fusion_adapter",
                "adapter_position": "post_fusion",
                "adapter_checkpoint": args.adapter_ckpt,
                "adapter_training_method": (
                    adapter_checkpoint or {}
                ).get("method", "unknown"),
                "ranking_score_type": "cosine_similarity_of_post_fusion_adapter_features",
            }
        )
    elif args.method == "lora":
        common.update(
            {
                "feature_source": "lora_adapted_clip_image_encoder",
                "lora_rgb_feat_root": args.lora_rgb_feat_root,
                "lora_depth_feat_root": args.lora_depth_feat_root,
                "ranking_score_type": "cosine_similarity_of_lora_fusion_features",
            }
        )
    elif args.method == "coop_seen_anchor":
        common.update(
            {
                "feature_source": "cached_base_clip_features",
                "text_anchor_scope": "seen",
                "uses_seen_class_names_as_anchors": True,
                "seen_anchor_weight": float(args.seen_anchor_weight),
                "seen_anchor_similarity": args.seen_anchor_similarity,
                "seen_anchor_power": float(args.seen_anchor_power),
                "prompt_ckpt": args.prompt_ckpt,
                "ranking_score_type": (
                    "(1-lambda)*fusion_visual_similarity + "
                    "lambda*seen_anchor_response_similarity"
                ),
            }
        )
    elif args.method == "cocoop_seen_anchor":
        common.update(
            {
                "feature_source": "cached_base_clip_features",
                "text_anchor_scope": "seen",
                "uses_seen_class_names_as_anchors": True,
                "seen_anchor_weight": float(args.seen_anchor_weight),
                "seen_anchor_similarity": args.seen_anchor_similarity,
                "seen_anchor_power": float(args.seen_anchor_power),
                "prompt_ckpt": args.prompt_ckpt,
                "ranking_score_type": (
                    "(1-lambda)*fusion_visual_similarity + "
                    "lambda*conditional_seen_anchor_response_similarity"
                ),
            }
        )

    return common


def main():
    args = parse_args()
    protocol = load_protocol(args.protocol)
    device = None
    if args.method in {"adapter", "coop_seen_anchor", "cocoop_seen_anchor"}:
        device = resolve_device(args.device)

    rgb_feat_root = args.rgb_feat_root
    depth_feat_root = args.depth_feat_root
    if args.method == "lora":
        rgb_feat_root, depth_feat_root = require_lora_feature_roots(args)

    gallery_visual, gallery_labels = load_fusion_split(
        protocol,
        "gallery_unseen",
        rgb_feat_root,
        depth_feat_root,
        args.alpha,
    )
    query_visual, query_labels = load_fusion_split(
        protocol,
        "query_unseen",
        rgb_feat_root,
        depth_feat_root,
        args.alpha,
    )

    adapter_checkpoint = None
    if args.method == "adapter":
        adapter, adapter_checkpoint = load_adapter(args.adapter_ckpt, device=device)
        gallery_visual = apply_adapter_to_features(
            gallery_visual,
            adapter,
            batch_size=args.batch_size,
        )
        query_visual = apply_adapter_to_features(
            query_visual,
            adapter,
            batch_size=args.batch_size,
        )

    print(f"Method: {args.method}")
    print(f"Protocol: {args.protocol}")
    print(f"RGB feature root: {rgb_feat_root}")
    print(f"Depth feature root: {depth_feat_root}")
    print(f"Gallery size: {gallery_visual.shape[0]}")
    print(f"Query size: {query_visual.shape[0]}")

    if args.method in SEEN_ANCHOR_METHODS:
        seen_classes = list(protocol["seen_classes"])
        validate_seen_anchor_classes(protocol, seen_classes, args.text_scope)
        print(f"Seen anchor count: {len(seen_classes)}")
        print(f"Seen anchor weight: {args.seen_anchor_weight}")

        if args.method == "coop_seen_anchor":
            gallery_response = build_coop_seen_response(
                gallery_visual,
                seen_classes,
                args.clip_model,
                args.prompt_ckpt,
                args.temperature,
                device,
            )
            query_response = build_coop_seen_response(
                query_visual,
                seen_classes,
                args.clip_model,
                args.prompt_ckpt,
                args.temperature,
                device,
            )
        else:
            gallery_response = build_cocoop_seen_response(
                gallery_visual,
                seen_classes,
                args.clip_model,
                args.prompt_ckpt,
                args.temperature,
                args.prompt_batch_size,
                args.prompt_chunk_size,
                device,
                desc="Building gallery CoCoOp seen-anchor response",
            )
            query_response = build_cocoop_seen_response(
                query_visual,
                seen_classes,
                args.clip_model,
                args.prompt_ckpt,
                args.temperature,
                args.prompt_batch_size,
                args.prompt_chunk_size,
                device,
                desc="Building query CoCoOp seen-anchor response",
            )

        sim_matrix = compute_seen_anchor_similarity(
            query_visual,
            gallery_visual,
            query_response,
            gallery_response,
            seen_anchor_weight=args.seen_anchor_weight,
            seen_anchor_similarity=args.seen_anchor_similarity,
            seen_anchor_power=args.seen_anchor_power,
            batch_size=args.batch_size,
        )
    else:
        sim_matrix = compute_similarity(
            query_visual,
            gallery_visual,
            batch_size=args.batch_size,
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

    os.makedirs(args.output_dir, exist_ok=True)
    save_name = args.save_name or default_save_name(args)
    save_path = os.path.join(args.output_dir, save_name)

    text_classes = list(protocol["seen_classes"]) if args.method in SEEN_ANCHOR_METHODS else []
    output = {
        "method": args.method,
        "protocol_path": args.protocol,
        "clip_model": args.clip_model,
        "seen_classes": protocol["seen_classes"],
        "unseen_classes": protocol["unseen_classes"],
        "text_classes": text_classes,
        "text_anchor_classes": text_classes,
        "gallery_size": int(gallery_visual.shape[0]),
        "query_size": int(query_visual.shape[0]),
        "rgb_feat_root": rgb_feat_root,
        "depth_feat_root": depth_feat_root,
        "metric_style": args.metric_style,
        "primary_metric_style": primary_style,
        "metrics": primary_metrics,
        "metrics_by_style": metrics_by_style,
    }
    output.update(method_metadata(args, adapter_checkpoint))
    if args.method in SEEN_ANCHOR_METHODS:
        output.update(
            {
                "seen_anchor_response_dim": int(len(protocol["seen_classes"])),
                "num_seen_text_anchors": int(len(protocol["seen_classes"])),
                "temperature": float(args.temperature),
                "prompt_batch_size": int(args.prompt_batch_size),
                "prompt_chunk_size": int(args.prompt_chunk_size),
            }
        )

    save_path = save_json_with_fallback(output, save_path)

    print(f"Saved: {save_path}")
    if args.metric_style == "both":
        print(f"{args.method} [hgm2r]: {format_metric_report(metrics_by_style['hgm2r'])}")
        print(f"{args.method} [legacy]: {format_metric_report(metrics_by_style['legacy'])}")
    else:
        print(f"{args.method} [{primary_style}]: {format_metric_report(primary_metrics)}")
    print("Safety: no unseen class names, unseen labels, or gallery labels used for ranking.")


if __name__ == "__main__":
    main()
