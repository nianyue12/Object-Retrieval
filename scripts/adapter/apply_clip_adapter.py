"""
功能：将训练好的共享 Adapter 应用到缓存好的 CLIP 特征目录。

说明：
    - 输入可以是单个 1D 特征，也可以是多视图 2D 特征
    - 输出仍然保存为 `.npy`，方便直接复用现有检索脚本
"""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.clip_adapter import CLIPResidualAdapter


def parse_args():
    # 这个脚本只做推理式特征转换，不再训练 Adapter。
    parser = argparse.ArgumentParser(
        description="Apply a trained residual Adapter to cached CLIP features."
    )
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    # 命令行未指定时自动选择 CUDA/CPU。
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str, device: torch.device):
    # weights_only 是新版 PyTorch 参数，旧版本不支持时回退到普通加载。
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_adapter(checkpoint: dict, device: torch.device) -> CLIPResidualAdapter:
    # 根据 checkpoint 中记录的结构参数重建同尺寸 Adapter。
    adapter = CLIPResidualAdapter(
        dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
        residual_scale=float(checkpoint.get("residual_scale", 0.2)),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    return adapter


def adapt_feature(
    adapter: CLIPResidualAdapter,
    feature: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 1:
        # 单个物体特征补 batch 维度，Adapter 输出后再挤掉。
        tensor = torch.from_numpy(feature).unsqueeze(0).to(device)
        squeeze = True
    elif feature.ndim == 2:
        # 多视图特征逐行适配，保持原来的视图维度。
        tensor = torch.from_numpy(feature).to(device)
        squeeze = False
    else:
        raise ValueError(f"Unsupported feature shape: {feature.shape}")

    with torch.no_grad():
        adapted = adapter(tensor).cpu().numpy().astype(np.float32)

    if squeeze:
        return adapted.squeeze(0)
    return adapted


def iter_feature_files(input_root: str):
    # 维持 `<class>/<item>.npy` 目录结构，方便输出与输入一一对应。
    for class_name in sorted(os.listdir(input_root)):
        class_dir = os.path.join(input_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in sorted(os.listdir(class_dir)):
            if not filename.endswith(".npy"):
                continue
            yield class_name, filename, os.path.join(class_dir, filename)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = safe_torch_load(args.adapter_ckpt, device=device)
    adapter = load_adapter(checkpoint, device=device)

    os.makedirs(args.output_root, exist_ok=True)
    saved_count = 0
    skipped_count = 0

    print(f"Loaded Adapter checkpoint: {args.adapter_ckpt}")
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Device: {device}")

    # 遍历所有缓存特征，逐个应用 Adapter 并写到新目录。
    for class_name, filename, input_path in iter_feature_files(args.input_root):
        class_output_dir = os.path.join(args.output_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        output_path = os.path.join(class_output_dir, filename)

        if os.path.exists(output_path) and not args.overwrite:
            skipped_count += 1
            continue

        feature = np.load(input_path).astype(np.float32)
        adapted = adapt_feature(adapter, feature, device=device)
        np.save(output_path, adapted)
        saved_count += 1

    print(f"Saved features: {saved_count}")
    print(f"Skipped existing features: {skipped_count}")


if __name__ == "__main__":
    main()
