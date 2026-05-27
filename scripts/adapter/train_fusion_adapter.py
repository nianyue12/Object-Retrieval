"""
Train a post-fusion residual Adapter on seen classes.

The training and evaluation path matches the revised thesis setting:

RGB feature + Depth feature -> Fusion feature -> Adapter -> seen-class loss.

At unseen retrieval time, the classifier is discarded and only the adapted
Fusion feature is used for query-gallery cosine ranking.
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import (  # noqa: E402
    ALPHA_FUSION,
    DEFAULT_PROTOCOL_PATH,
    DEPTH_FEAT_DIR,
    RESULT_DIR,
    RGB_FEAT_DIR,
)
from models.clip_adapter import CLIPResidualAdapter  # noqa: E402
from utils.clip_utils import load_clip_model  # noqa: E402
from utils.features import load_feature  # noqa: E402
from utils.protocol import get_split_items, load_protocol  # noqa: E402
from utils.semantic import PROMPT_TEMPLATES  # noqa: E402


DEFAULT_SAVE_DIR = os.path.join(RESULT_DIR, "adapter")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a residual Adapter after RGB+Depth Fusion features."
    )
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--alpha", type=float, default=ALPHA_FUSION)
    parser.add_argument("--rgb_feat_root", type=str, default=RGB_FEAT_DIR)
    parser.add_argument("--depth_feat_root", type=str, default=DEPTH_FEAT_DIR)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual_scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--save_name", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_rows(feats: np.ndarray) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.clip(norms, 1e-12, None)


def fuse_features(
    rgb_feats: np.ndarray,
    depth_feats: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return normalize_rows(alpha * rgb_feats + (1.0 - alpha) * depth_feats)


def load_split_fusion_features(
    protocol: dict,
    split_name: str,
    rgb_feat_root: str,
    depth_feat_root: str,
    label_to_index: dict,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb_feats = []
    depth_feats = []
    labels = []

    for cls, item in get_split_items(protocol, split_name):
        rgb_path = os.path.join(rgb_feat_root, cls, item)
        depth_path = os.path.join(depth_feat_root, cls, item)
        rgb_feats.append(load_feature(rgb_path, aggregation="mean"))
        depth_feats.append(load_feature(depth_path, aggregation="mean"))
        labels.append(label_to_index[cls])

    fusion_feats = fuse_features(
        normalize_rows(np.stack(rgb_feats).astype(np.float32)),
        normalize_rows(np.stack(depth_feats).astype(np.float32)),
        alpha,
    )
    return fusion_feats, np.array(labels, dtype=np.int64)


def build_loader(
    feats: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(feats), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_text_prototypes_and_logit_scale(
    class_names,
    clip_model_name: str,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    clip_module, clip_model, _ = load_clip_model(
        clip_model_name,
        device=device,
        force_float=True,
    )
    clip_model.eval()

    prototypes = []
    with torch.no_grad():
        for class_name in class_names:
            readable_name = class_name.replace("_", " ")
            prompts = [template.format(readable_name) for template in PROMPT_TEMPLATES]
            tokenized = clip_module.tokenize(prompts).to(device)
            text_features = clip_model.encode_text(tokenized)
            text_features = F.normalize(text_features, dim=-1)
            pooled = F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1)
            prototypes.append(pooled.squeeze(0))

        logit_scale = float(clip_model.logit_scale.exp().clamp(max=100.0).item())

    text_bank = torch.stack(prototypes, dim=0)
    text_bank = F.normalize(text_bank, dim=-1)

    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return text_bank, logit_scale


def compute_logits(
    adapted_feats: torch.Tensor,
    text_prototypes: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    return logit_scale * adapted_feats @ text_prototypes.t()


def run_epoch(
    loader: DataLoader,
    adapter: CLIPResidualAdapter,
    text_prototypes: torch.Tensor,
    logit_scale: float,
    optimizer=None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    adapter.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_feats, batch_labels in loader:
        batch_feats = batch_feats.to(text_prototypes.device, non_blocking=True)
        batch_labels = batch_labels.to(text_prototypes.device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        adapted_feats = adapter(batch_feats)
        logits = compute_logits(adapted_feats, text_prototypes, logit_scale)
        loss = F.cross_entropy(logits, batch_labels)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(batch_labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
        total_count += batch_size

    return {
        "loss": total_loss / max(1, total_count),
        "top1": total_correct / max(1, total_count),
    }


def build_default_save_name(args) -> str:
    return f"fusion_post_adapter_h{args.hidden_dim}_seed{args.seed}.pt"


def checkpoint_state_dict(module: torch.nn.Module) -> dict:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def save_checkpoint(path: str, checkpoint: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    protocol = load_protocol(args.protocol)
    seen_classes = list(protocol["seen_classes"])
    label_to_index = {cls: idx for idx, cls in enumerate(seen_classes)}

    train_feats, train_labels = load_split_fusion_features(
        protocol,
        "train_seen",
        args.rgb_feat_root,
        args.depth_feat_root,
        label_to_index,
        args.alpha,
    )
    val_feats, val_labels = load_split_fusion_features(
        protocol,
        "val_seen",
        args.rgb_feat_root,
        args.depth_feat_root,
        label_to_index,
        args.alpha,
    )

    device = resolve_device(args.device)
    text_prototypes, logit_scale = build_text_prototypes_and_logit_scale(
        seen_classes,
        clip_model_name=args.clip_model,
        device=device,
    )

    adapter = CLIPResidualAdapter(
        dim=int(train_feats.shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)

    train_loader = build_loader(
        train_feats,
        train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        val_feats,
        val_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_top1 = -1.0
    best_epoch = -1
    history = []
    save_path = os.path.join(
        args.save_dir,
        args.save_name or build_default_save_name(args),
    )

    print(f"Seen classes: {seen_classes}")
    print("Adapter position: post_fusion")
    print(f"Train size: {len(train_labels)}")
    print(f"Val size: {len(val_labels)}")
    print(f"Device: {device}")
    print(f"Feature dim: {train_feats.shape[1]}")
    print(f"Save path: {save_path}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader,
            adapter,
            text_prototypes,
            logit_scale,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                val_loader,
                adapter,
                text_prototypes,
                logit_scale,
                optimizer=None,
            )

        history_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
        }
        history.append(history_entry)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_top1={train_metrics['top1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.4f}"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            checkpoint = {
                "method": "fusion_post_adapter",
                "clip_model": args.clip_model,
                "protocol_path": args.protocol,
                "adapter_position": "post_fusion",
                "base_feature": "rgb_depth_fusion",
                "alpha": float(args.alpha),
                "seed": int(args.seed),
                "feature_dim": int(train_feats.shape[1]),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "residual_scale": float(args.residual_scale),
                "logit_scale": float(logit_scale),
                "seen_classes": seen_classes,
                "train_size": int(len(train_labels)),
                "val_size": int(len(val_labels)),
                "best_epoch": int(best_epoch),
                "best_val_top1": float(best_val_top1),
                "uses_unseen_class_names_for_training": False,
                "uses_unseen_labels_for_training": False,
                "adapter_state_dict": checkpoint_state_dict(adapter),
                "history": history,
            }
            save_checkpoint(save_path, checkpoint)

    summary_path = os.path.splitext(save_path)[0] + ".json"
    summary = {
        "save_path": save_path,
        "method": "fusion_post_adapter",
        "clip_model": args.clip_model,
        "protocol_path": args.protocol,
        "adapter_position": "post_fusion",
        "base_feature": "rgb_depth_fusion",
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "feature_dim": int(train_feats.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "residual_scale": float(args.residual_scale),
        "logit_scale": float(logit_scale),
        "best_epoch": int(best_epoch),
        "best_val_top1": float(best_val_top1),
        "train_size": int(len(train_labels)),
        "val_size": int(len(val_labels)),
        "uses_unseen_class_names_for_training": False,
        "uses_unseen_labels_for_training": False,
        "history": history,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Best post-fusion Adapter checkpoint saved to: {save_path}")
    print(f"Best val top1: {best_val_top1:.4f} at epoch {best_epoch}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
