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
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_adapter(checkpoint: dict, device: torch.device) -> CLIPResidualAdapter:
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
        tensor = torch.from_numpy(feature).unsqueeze(0).to(device)
        squeeze = True
    elif feature.ndim == 2:
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
