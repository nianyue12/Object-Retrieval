import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import BASE_DIR
from utils.clip_utils import load_clip_model
from utils.lora import apply_lora_to_clip, load_lora_state_dict

RGB_VIEW_ROOT = os.path.join(BASE_DIR, "output_224")
DEPTH_MAP_ROOT = os.path.join(BASE_DIR, "depth_maps")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP features with a trained visual LoRA checkpoint."
    )
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument("--modality", choices=["rgb", "depth"], required=True)
    parser.add_argument("--input_root", type=str, default="")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_depth_image(path: str) -> Image.Image:
    depth = Image.open(path)
    depth_array = np.array(depth, dtype=np.float32)

    if depth_array.ndim == 3:
        depth_array = depth_array[..., 0]

    if depth_array.max() > depth_array.min():
        depth_array = (depth_array - depth_array.min()) / (
            depth_array.max() - depth_array.min()
        )
    else:
        depth_array = np.zeros_like(depth_array)

    depth_array = (depth_array * 255).astype(np.uint8)
    depth_array = np.stack([depth_array] * 3, axis=-1)
    return Image.fromarray(depth_array, mode="RGB")


def iter_rgb_objects(input_root: str) -> List[Tuple[str, str, List[str]]]:
    objects = []
    for category_dir in sorted(os.listdir(input_root)):
        full_category_dir = os.path.join(input_root, category_dir)
        if not os.path.isdir(full_category_dir):
            continue
        if not category_dir.endswith("_multi_view"):
            continue

        class_name = category_dir.replace("_multi_view", "")
        for object_id in sorted(os.listdir(full_category_dir)):
            object_dir = os.path.join(full_category_dir, object_id)
            if not os.path.isdir(object_dir):
                continue
            view_paths = []
            for view_idx in range(12):
                view_path = os.path.join(object_dir, f"rgb_{view_idx:04d}.png")
                if os.path.exists(view_path):
                    view_paths.append(view_path)
            objects.append((class_name, object_id, view_paths))
    return objects


def iter_depth_objects(input_root: str) -> List[Tuple[str, str, List[str]]]:
    objects = []
    for class_name in sorted(os.listdir(input_root)):
        class_dir = os.path.join(input_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for object_id in sorted(os.listdir(class_dir)):
            object_dir = os.path.join(class_dir, object_id)
            if not os.path.isdir(object_dir):
                continue
            view_paths = []
            for view_idx in range(12):
                view_path = os.path.join(object_dir, f"depth_{view_idx:02d}.png")
                if os.path.exists(view_path):
                    view_paths.append(view_path)
            objects.append((class_name, object_id, view_paths))
    return objects


def encode_object_views(
    model,
    preprocess,
    view_paths: List[str],
    modality: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not view_paths:
        raise ValueError("view_paths must contain at least one file.")

    tensors = []
    for view_path in view_paths:
        if modality == "rgb":
            image = load_rgb_image(view_path)
        else:
            image = load_depth_image(view_path)
        tensors.append(preprocess(image))

    all_views = torch.stack(tensors, dim=0)
    encoded = []
    with torch.no_grad():
        for start in range(0, all_views.shape[0], batch_size):
            end = min(start + batch_size, all_views.shape[0])
            batch = all_views[start:end].to(device)
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            encoded.append(features.cpu())

    view_features = torch.cat(encoded, dim=0).numpy().astype(np.float32)
    pooled = view_features.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = safe_torch_load(args.lora_ckpt, device)

    input_root = args.input_root
    if not input_root:
        input_root = RGB_VIEW_ROOT if args.modality == "rgb" else DEPTH_MAP_ROOT

    _, model, preprocess = load_clip_model(
        checkpoint["clip_model"],
        device=device,
        force_float=True,
    )
    apply_lora_to_clip(
        model,
        rank=int(checkpoint["rank"]),
        alpha=float(checkpoint["lora_alpha"]),
        dropout=float(checkpoint.get("lora_dropout", 0.0)),
        block_indices=checkpoint["visual_block_indices"],
        module_suffixes=checkpoint["module_suffixes"],
    )
    load_lora_state_dict(model, checkpoint["lora_state_dict"], strict=True)
    model.eval()

    if args.modality == "rgb":
        objects = iter_rgb_objects(input_root)
    else:
        objects = iter_depth_objects(input_root)

    os.makedirs(args.output_root, exist_ok=True)
    saved_count = 0
    skipped_count = 0

    print(f"Loaded LoRA checkpoint: {args.lora_ckpt}")
    print(f"Modality: {args.modality}")
    print(f"Input root: {input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Object count: {len(objects)}")

    for class_name, object_id, view_paths in tqdm(objects, desc="Extracting LoRA features"):
        if not view_paths:
            skipped_count += 1
            continue

        class_output_dir = os.path.join(args.output_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        output_path = os.path.join(class_output_dir, f"{object_id}.npy")

        if os.path.exists(output_path) and not args.overwrite:
            skipped_count += 1
            continue

        pooled_feature = encode_object_views(
            model=model,
            preprocess=preprocess,
            view_paths=view_paths,
            modality=args.modality,
            device=device,
            batch_size=args.batch_size,
        )
        np.save(output_path, pooled_feature)
        saved_count += 1

    print(f"Saved features: {saved_count}")
    print(f"Skipped objects: {skipped_count}")


if __name__ == "__main__":
    main()
