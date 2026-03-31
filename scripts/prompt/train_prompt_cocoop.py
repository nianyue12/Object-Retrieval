import argparse
import json
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import (
    ALPHA_FUSION,
    DEFAULT_PROTOCOL_PATH,
    DEPTH_FEAT_DIR,
    RESULT_DIR,
    RGB_FEAT_DIR,
)
from models.cocoop_prompt_learner import ConditionalPromptLearner
from models.prompt_learner import TextEncoder
from utils.clip_utils import load_clip_model
from utils.features import load_feature
from utils.protocol import get_split_items, load_protocol


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CoCoOp prompts on cached CLIP features."
    )
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--mode", choices=["rgb", "depth", "fusion"], default="fusion")
    parser.add_argument("--alpha", type=float, default=ALPHA_FUSION)
    parser.add_argument("--rgb_feat_root", type=str, default=RGB_FEAT_DIR)
    parser.add_argument("--depth_feat_root", type=str, default=DEPTH_FEAT_DIR)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--prompt_chunk_size",
        type=int,
        default=128,
        help="How many flattened prompts to encode at once inside the text branch.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_ctx", type=int, default=8)
    parser.add_argument("--ctx_init", type=str, default="")
    parser.add_argument("--meta_hidden_dim", type=int, default=64)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable CUDA AMP mixed precision.",
    )
    parser.add_argument(
        "--class_token_position",
        choices=["end"],
        default="end",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(RESULT_DIR, "prompt_tuning"),
    )
    parser.add_argument("--save_name", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(feats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.clip(norms, 1e-12, None)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(use_amp: bool):
    try:
        return torch.amp.autocast("cuda", enabled=use_amp)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=use_amp)


def load_split_features(
    protocol: dict,
    split_name: str,
    mode: str,
    alpha: float,
    rgb_feat_root: str,
    depth_feat_root: str,
    label_to_index: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []

    for cls, item in get_split_items(protocol, split_name):
        rgb_path = os.path.join(rgb_feat_root, cls, item)
        depth_path = os.path.join(depth_feat_root, cls, item)

        if mode == "rgb":
            feat = load_feature(rgb_path, aggregation="mean")
        elif mode == "depth":
            feat = load_feature(depth_path, aggregation="mean")
        else:
            rgb_feat = load_feature(rgb_path, aggregation="mean")
            depth_feat = load_feature(depth_path, aggregation="mean")
            feat = alpha * rgb_feat + (1.0 - alpha) * depth_feat
            feat = feat / np.clip(np.linalg.norm(feat), 1e-12, None)

        feats.append(feat.astype(np.float32))
        labels.append(label_to_index[cls])

    return normalize_rows(np.stack(feats)), np.array(labels, dtype=np.int64)


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
    )


def compute_logits(
    image_feats: torch.Tensor,
    prompt_learner: ConditionalPromptLearner,
    text_encoder: TextEncoder,
    temperature: torch.Tensor,
    prompt_chunk_size: int,
) -> torch.Tensor:
    image_features = F.normalize(image_feats, dim=-1)
    text_features = prompt_learner.get_text_features(
        image_features,
        text_encoder,
        prompt_chunk_size=prompt_chunk_size,
    )
    logits = torch.einsum("bd,bcd->bc", image_features, text_features)
    return logits / temperature


def run_epoch(
    loader: DataLoader,
    prompt_learner: ConditionalPromptLearner,
    text_encoder: TextEncoder,
    temperature: torch.Tensor,
    prompt_chunk_size: int,
    use_amp: bool,
    scaler=None,
    desc: str = "",
    optimizer=None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    prompt_learner.train(is_train)
    text_encoder.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for batch_feats, batch_labels in progress:
        batch_feats = batch_feats.to(prompt_learner.ctx.device)
        batch_labels = batch_labels.to(prompt_learner.ctx.device)

        if is_train:
            optimizer.zero_grad()

        with autocast_context(use_amp):
            logits = compute_logits(
                batch_feats,
                prompt_learner,
                text_encoder,
                temperature,
                prompt_chunk_size,
            )
            loss = F.cross_entropy(logits, batch_labels)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * batch_labels.size(0)
        total_correct += int((preds == batch_labels).sum().item())
        total_count += int(batch_labels.size(0))

        avg_loss = total_loss / max(1, total_count)
        avg_acc = total_correct / max(1, total_count)
        progress.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_acc:.4f}")

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


def build_default_save_name(args) -> str:
    ctx_tag = f"nctx{args.n_ctx}"
    init_tag = args.ctx_init.strip().lower().replace(" ", "_")
    if not init_tag:
        init_tag = "random"
    init_tag = "".join(ch for ch in init_tag if ch.isalnum() or ch == "_")
    return f"cocoop_{args.mode}_{ctx_tag}_{init_tag}_seed{args.seed}.pt"


def save_checkpoint(path: str, checkpoint: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    protocol = load_protocol(args.protocol)
    seen_classes = list(protocol["seen_classes"])
    label_to_index = {cls: idx for idx, cls in enumerate(seen_classes)}

    train_feats, train_labels = load_split_features(
        protocol,
        "train_seen",
        args.mode,
        args.alpha,
        args.rgb_feat_root,
        args.depth_feat_root,
        label_to_index,
    )
    val_feats, val_labels = load_split_features(
        protocol,
        "val_seen",
        args.mode,
        args.alpha,
        args.rgb_feat_root,
        args.depth_feat_root,
        label_to_index,
    )

    device = resolve_device(args.device)
    _, clip_model, _ = load_clip_model(
        args.clip_model,
        device=device,
        force_float=True,
    )
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    prompt_learner = ConditionalPromptLearner(
        class_names=seen_classes,
        clip_model=clip_model,
        image_feature_dim=int(train_feats.shape[1]),
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        meta_hidden_dim=args.meta_hidden_dim,
        class_token_position=args.class_token_position,
    ).to(device)
    text_encoder = TextEncoder(clip_model).to(device)
    temperature = (1.0 / clip_model.logit_scale.exp().detach()).to(device)
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = make_grad_scaler(use_amp)

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
        prompt_learner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_epoch = -1
    history = []
    save_path = os.path.join(
        args.save_dir,
        args.save_name or build_default_save_name(args),
    )

    print(f"Seen classes: {seen_classes}")
    print(f"Train size: {len(train_labels)}")
    print(f"Val size: {len(val_labels)}")
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Eval every: {args.eval_every}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            train_loader,
            prompt_learner,
            text_encoder,
            temperature,
            args.prompt_chunk_size,
            use_amp,
            scaler=scaler,
            desc=f"Train {epoch:03d}",
            optimizer=optimizer,
        )
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            with torch.no_grad():
                val_loss, val_acc = run_epoch(
                    val_loader,
                    prompt_learner,
                    text_encoder,
                    temperature,
                    args.prompt_chunk_size,
                    use_amp,
                    scaler=None,
                    desc=f"Val {epoch:03d}",
                    optimizer=None,
                )
        else:
            val_loss, val_acc = float("nan"), float("nan")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_acc,
                "val_loss": val_loss,
                "val_top1": val_acc,
            }
        )

        if should_eval:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_top1={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_top1={val_acc:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_top1={train_acc:.4f} | "
                f"val=skipped"
            )

        if should_eval and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "prompt_mode": "cocoop",
                "clip_model": args.clip_model,
                "mode": args.mode,
                "alpha": float(args.alpha),
                "protocol_path": args.protocol,
                "n_ctx": int(args.n_ctx),
                "ctx_init": args.ctx_init,
                "class_token_position": args.class_token_position,
                "seed": int(args.seed),
                "meta_hidden_dim": int(args.meta_hidden_dim),
                "prompt_chunk_size": int(args.prompt_chunk_size),
                "eval_every": int(args.eval_every),
                "amp_enabled": bool(use_amp),
                "feature_dim": int(train_feats.shape[1]),
                "seen_classes": seen_classes,
                "train_size": int(len(train_labels)),
                "val_size": int(len(val_labels)),
                "best_epoch": int(best_epoch),
                "best_val_top1": float(best_val_acc),
                "ctx": prompt_learner.get_context().cpu(),
                "meta_net_state_dict": {
                    key: value.detach().cpu()
                    for key, value in prompt_learner.meta_net.state_dict().items()
                },
                "history": history,
            }
            save_checkpoint(save_path, checkpoint)

    summary_path = os.path.splitext(save_path)[0] + ".json"
    summary = {
        "save_path": save_path,
        "prompt_mode": "cocoop",
        "clip_model": args.clip_model,
        "mode": args.mode,
        "alpha": float(args.alpha),
        "protocol_path": args.protocol,
        "n_ctx": int(args.n_ctx),
        "ctx_init": args.ctx_init,
        "class_token_position": args.class_token_position,
        "seed": int(args.seed),
        "meta_hidden_dim": int(args.meta_hidden_dim),
        "prompt_chunk_size": int(args.prompt_chunk_size),
        "eval_every": int(args.eval_every),
        "amp_enabled": bool(use_amp),
        "best_epoch": int(best_epoch),
        "best_val_top1": float(best_val_acc),
        "train_size": int(len(train_labels)),
        "val_size": int(len(val_labels)),
        "history": history,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Best checkpoint saved to: {save_path}")
    print(f"Best val top1: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
