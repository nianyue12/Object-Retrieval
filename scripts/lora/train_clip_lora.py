"""
功能：在缓存好的 RGB / Depth 视图上训练 CLIP 视觉分支的 LoRA 适配器。

说明：
    训练目标是 seen 类分类，
    最终保存 LoRA 参数，供后续特征提取或检索脚本使用。
"""

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import BASE_DIR, DEFAULT_PROTOCOL_PATH, RESULT_DIR
from utils.clip_utils import get_clip_module, load_clip_model
from utils.lora import (
    apply_lora_to_clip,
    count_all_parameters,
    count_trainable_parameters,
    extract_lora_state_dict,
    mark_only_lora_trainable,
    parse_int_list,
    parse_str_list,
)
from utils.protocol import get_split_items, load_protocol
from utils.semantic import PROMPT_TEMPLATES

RGB_VIEW_ROOT = os.path.join(BASE_DIR, "output_224")
DEPTH_MAP_ROOT = os.path.join(BASE_DIR, "depth_maps")
DEFAULT_SAVE_DIR = os.path.join(RESULT_DIR, "lora")


def parse_args():
    """解析训练脚本命令行参数。"""
    # LoRA 训练直接读原始 RGB/Depth 视图，因为它会修改 CLIP 视觉编码过程。
    parser = argparse.ArgumentParser(
        description="Train a visual LoRA adapter on CLIP for RGB/depth view classification."
    )
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--mode", choices=["rgb", "depth", "fusion"], default="fusion")
    parser.add_argument("--rgb_root", type=str, default=RGB_VIEW_ROOT)
    parser.add_argument("--depth_root", type=str, default=DEPTH_MAP_ROOT)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--visual_blocks",
        type=str,
        default="8,9,10,11",
        help="Comma-separated CLIP visual transformer block indices to adapt.",
    )
    parser.add_argument(
        "--lora_modules",
        type=str,
        default="mlp.c_fc,mlp.c_proj,attn.out_proj",
        help="Comma-separated suffixes of visual modules to wrap with LoRA.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--save_name", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验更可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """根据参数或环境自动选择设备。"""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grad_scaler(use_amp: bool):
    """创建 AMP 梯度缩放器，兼容不同 PyTorch 版本。"""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(use_amp: bool):
    """返回自动混合精度上下文。"""
    if not use_amp:
        return nullcontext()

    try:
        return torch.amp.autocast("cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def load_rgb_image(path: str) -> Image.Image:
    """读取 RGB 图像。"""
    return Image.open(path).convert("RGB")


def load_depth_image(path: str) -> Image.Image:
    """读取深度图并转换成 3 通道 RGB 形式。"""
    depth = Image.open(path)
    depth_array = np.array(depth, dtype=np.float32)

    if depth_array.ndim == 3:
        depth_array = depth_array[..., 0]

    # 深度图先归一化到 [0, 1]，再复制成 3 通道以适配 CLIP 图像预处理。
    if depth_array.max() > depth_array.min():
        depth_array = (depth_array - depth_array.min()) / (
            depth_array.max() - depth_array.min()
        )
    else:
        depth_array = np.zeros_like(depth_array)

    depth_array = (depth_array * 255).astype(np.uint8)
    depth_array = np.stack([depth_array] * 3, axis=-1)
    return Image.fromarray(depth_array, mode="RGB")


def build_view_samples(
    protocol: dict,
    split_name: str,
    mode: str,
    rgb_root: str,
    depth_root: str,
    label_to_index: dict,
) -> Tuple[List[Tuple[str, int, str]], dict]:
    """
    功能：把协议里的物体样本展开成“单视图训练样本”列表。

    返回：
        samples: `(image_path, label_id, modality)` 列表
        stats: 统计信息，方便打印数据规模
    """
    samples = []
    stats = {
        "objects": 0,
        "rgb_views": 0,
        "depth_views": 0,
        "missing_objects": 0,
    }

    # 一个物体会展开成多个视图样本，LoRA 用视图级分类信号训练视觉分支。
    for cls, item in get_split_items(protocol, split_name):
        obj_id = os.path.splitext(item)[0]
        added = 0
        stats["objects"] += 1

        if mode in {"rgb", "fusion"}:
            rgb_obj_dir = os.path.join(rgb_root, f"{cls}_multi_view", obj_id)
            # RGB 分支读取多视图渲染图
            for view_idx in range(12):
                view_path = os.path.join(rgb_obj_dir, f"rgb_{view_idx:04d}.png")
                if not os.path.exists(view_path):
                    continue
                samples.append((view_path, label_to_index[cls], "rgb"))
                stats["rgb_views"] += 1
                added += 1

        if mode in {"depth", "fusion"}:
            depth_obj_dir = os.path.join(depth_root, cls, obj_id)
            # 深度分支读取深度图
            for view_idx in range(12):
                view_path = os.path.join(depth_obj_dir, f"depth_{view_idx:02d}.png")
                if not os.path.exists(view_path):
                    continue
                samples.append((view_path, label_to_index[cls], "depth"))
                stats["depth_views"] += 1
                added += 1

        if added == 0:
            stats["missing_objects"] += 1

    return samples, stats


class ViewDataset(Dataset):
    """把视图路径列表包装成可供 DataLoader 使用的数据集。"""

    def __init__(self, samples: Sequence[Tuple[str, int, str]], preprocess):
        self.samples = list(samples)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, modality = self.samples[index]
        if modality == "rgb":
            image = load_rgb_image(image_path)
        else:
            image = load_depth_image(image_path)
        # 返回预处理后的图像张量和类别标签
        return self.preprocess(image), label


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int):
    """构建训练或验证用的 DataLoader。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_text_features(
    class_names: Sequence[str],
    model,
    device: torch.device,
) -> torch.Tensor:
    """
    功能：为所有 seen 类构建固定文本原型。
    """
    clip_module = get_clip_module()
    text_features = []

    with torch.no_grad():
        for class_name in class_names:
            readable_name = class_name.replace("_", " ")
            # 同一类别使用多模板描述，再求平均
            prompts = [
                template.format(readable_name) for template in PROMPT_TEMPLATES
            ]
            tokenized = clip_module.tokenize(prompts).to(device)
            class_features = model.encode_text(tokenized)
            class_features = F.normalize(class_features, dim=-1)
            pooled = F.normalize(class_features.mean(dim=0, keepdim=True), dim=-1)
            text_features.append(pooled.squeeze(0))

    return torch.stack(text_features, dim=0)


def compute_logits(images, model, text_features):
    """计算图像特征与文本原型之间的分类 logits。"""
    image_features = model.encode_image(images)
    image_features = F.normalize(image_features, dim=-1)
    # 文本原型固定，训练信号只通过图像编码器中的 LoRA 参数回传。
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    return logit_scale * image_features @ text_features.t()


def run_epoch(
    loader,
    model,
    text_features,
    device,
    use_amp: bool,
    optimizer=None,
    scaler=None,
    desc: str = "",
):
    """
    功能：执行一轮训练或验证。

    说明：
        optimizer 不为空时走训练模式，否则走验证模式。
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # 训练时 optimizer/scaler 不为空；验证时只前向计算指标。
    progress = tqdm(loader, desc=desc, leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with autocast_context(use_amp):
            logits = compute_logits(images, model, text_features)
            loss = F.cross_entropy(logits, labels)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((preds == labels).sum().item())
        total_count += int(labels.size(0))

        avg_loss = total_loss / max(1, total_count)
        avg_acc = total_correct / max(1, total_count)
        progress.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_acc:.4f}")

    return (
        total_loss / max(1, total_count),
        total_correct / max(1, total_count),
    )


def build_default_save_name(args) -> str:
    """根据主要超参数生成默认保存名。"""
    block_tag = "-".join(str(idx) for idx in parse_int_list(args.visual_blocks))
    return (
        f"clip_lora_{args.mode}_r{args.rank}_a{int(args.lora_alpha)}"
        f"_b{block_tag}_seed{args.seed}.pt"
    )


def save_checkpoint(path: str, checkpoint: dict) -> None:
    """保存训练得到的 checkpoint。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def main():
    """脚本入口：准备数据、训练 LoRA、保存最佳结果。"""
    args = parse_args()
    set_seed(args.seed)

    block_indices = parse_int_list(args.visual_blocks)
    module_suffixes = parse_str_list(args.lora_modules)
    protocol = load_protocol(args.protocol)
    seen_classes = list(protocol["seen_classes"])
    label_to_index = {cls: idx for idx, cls in enumerate(seen_classes)}

    train_samples, train_stats = build_view_samples(
        protocol=protocol,
        split_name="train_seen",
        mode=args.mode,
        rgb_root=args.rgb_root,
        depth_root=args.depth_root,
        label_to_index=label_to_index,
    )
    val_samples, val_stats = build_view_samples(
        protocol=protocol,
        split_name="val_seen",
        mode=args.mode,
        rgb_root=args.rgb_root,
        depth_root=args.depth_root,
        label_to_index=label_to_index,
    )

    if not train_samples:
        raise RuntimeError("No training views found. Check rgb_root/depth_root and protocol.")
    if not val_samples:
        raise RuntimeError("No validation views found. Check rgb_root/depth_root and protocol.")

    device = resolve_device(args.device)
    # 先加载基础 CLIP，再把指定模块替换成 LoRA 版本
    _, model, preprocess = load_clip_model(
        args.clip_model,
        device=device,
        force_float=True,
    )
    model.eval()
    replaced_modules = apply_lora_to_clip(
        model,
        rank=args.rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        block_indices=block_indices,
        module_suffixes=module_suffixes,
    )
    trainable_param_names = mark_only_lora_trainable(model)

    # 注入 LoRA 后，只把 LoRA 参数交给优化器，其余 CLIP 权重保持冻结。
    # 构建 seen 类视图分类数据集
    train_dataset = ViewDataset(train_samples, preprocess=preprocess)
    val_dataset = ViewDataset(val_samples, preprocess=preprocess)
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 文本原型在训练过程中固定不变，只计算一次
    text_features = build_text_features(seen_classes, model, device=device)
    text_features = text_features.detach()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = make_grad_scaler(use_amp)

    best_val_acc = -1.0
    best_epoch = -1
    history = []
    save_path = os.path.join(
        args.save_dir,
        args.save_name or build_default_save_name(args),
    )

    print(f"Seen classes: {seen_classes}")
    print(f"Train samples: {len(train_dataset)} | stats: {train_stats}")
    print(f"Val samples: {len(val_dataset)} | stats: {val_stats}")
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Replaced LoRA modules ({len(replaced_modules)}): {replaced_modules}")
    print(
        "Trainable params: "
        f"{count_trainable_parameters(model):,} / {count_all_parameters(model):,}"
    )

    # 每轮训练后在 seen 验证集上选最优 checkpoint
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            train_loader,
            model,
            text_features,
            device,
            use_amp=use_amp,
            optimizer=optimizer,
            scaler=scaler,
            desc=f"Train {epoch:03d}",
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                val_loader,
                model,
                text_features,
                device,
                use_amp=use_amp,
                optimizer=None,
                scaler=None,
                desc=f"Val {epoch:03d}",
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_acc,
                "val_loss": val_loss,
                "val_top1": val_acc,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_top1={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_top1={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "method": "clip_visual_lora",
                "clip_model": args.clip_model,
                "protocol_path": args.protocol,
                "mode": args.mode,
                "seed": int(args.seed),
                "rank": int(args.rank),
                "lora_alpha": float(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "visual_block_indices": block_indices,
                "module_suffixes": module_suffixes,
                "seen_classes": seen_classes,
                "train_sample_count": int(len(train_dataset)),
                "val_sample_count": int(len(val_dataset)),
                "train_stats": train_stats,
                "val_stats": val_stats,
                "best_epoch": int(best_epoch),
                "best_val_top1": float(best_val_acc),
                "replaced_modules": replaced_modules,
                "trainable_param_names": trainable_param_names,
                "lora_state_dict": extract_lora_state_dict(model),
                "history": history,
            }
            save_checkpoint(save_path, checkpoint)

    summary_path = os.path.splitext(save_path)[0] + ".json"
    summary = {
        "save_path": save_path,
        "method": "clip_visual_lora",
        "clip_model": args.clip_model,
        "protocol_path": args.protocol,
        "mode": args.mode,
        "seed": int(args.seed),
        "rank": int(args.rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "visual_block_indices": block_indices,
        "module_suffixes": module_suffixes,
        "seen_classes": seen_classes,
        "train_sample_count": int(len(train_dataset)),
        "val_sample_count": int(len(val_dataset)),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "best_epoch": int(best_epoch),
        "best_val_top1": float(best_val_acc),
        "replaced_modules": replaced_modules,
        "trainable_param_count": int(count_trainable_parameters(model)),
        "all_param_count": int(count_all_parameters(model)),
        "history": history,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Best LoRA checkpoint saved to: {save_path}")
    print(f"Best val top1: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
