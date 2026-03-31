import argparse
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import BASE_DIR
from models.clip_encoder import CLIPEncoder
from utils.protocol import load_protocol

DATA_ROOT = os.path.join(BASE_DIR, "OS_MN40_core")
DEPTH_ROOT = os.path.join(BASE_DIR, "OS_MN40_core_depth_maps")
DEFAULT_PROTOCOL_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "splits",
    "os_mn40_core_seen8_unseen32_seed0.json",
)
DEFAULT_RGB_OUT = os.path.join(BASE_DIR, "OS_MN40_core_rgb_clip_feat")
DEFAULT_DEPTH_OUT = os.path.join(BASE_DIR, "OS_MN40_core_depth_clip_feat")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP features for OS-MN40-core RGB views or depth maps."
    )
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--modality", choices=["rgb", "depth"], required=True)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--depth_root", type=str, default=DEPTH_ROOT)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_root(args) -> str:
    if args.output_root:
        return args.output_root
    if args.modality == "rgb":
        return DEFAULT_RGB_OUT
    return DEFAULT_DEPTH_OUT


def build_unique_items(protocol: dict) -> List[Tuple[str, str, str]]:
    unique = {}

    for class_name, items in protocol["train_seen"].items():
        for item in items:
            unique[(class_name, item)] = "train_seen"

    for class_name, items in protocol["val_seen"].items():
        for item in items:
            unique[(class_name, item)] = "val_seen"

    for class_name, items in protocol["gallery_unseen"].items():
        for item in items:
            unique[(class_name, item)] = "gallery_unseen"

    for class_name, items in protocol["query_unseen"].items():
        for item in items:
            unique[(class_name, item)] = "query_unseen"

    return [(class_name, item, split_name) for (class_name, item), split_name in unique.items()]


def parse_angle(filename: str) -> int:
    name = os.path.splitext(filename)[0]
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def list_rgb_views(image_dir: str) -> List[str]:
    view_paths = [
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(".jpg")
    ]
    view_paths.sort(key=lambda path: parse_angle(os.path.basename(path)))
    return view_paths


def list_depth_views(depth_dir: str) -> List[str]:
    view_paths = [
        os.path.join(depth_dir, filename)
        for filename in os.listdir(depth_dir)
        if filename.lower().startswith("depth_") and filename.lower().endswith(".png")
    ]
    view_paths.sort()
    return view_paths


def load_depth_rgb(path: str) -> Image.Image:
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


def resolve_rgb_dir(data_root: str, split_name: str, class_name: str, object_id: str) -> str:
    if split_name in {"train_seen", "val_seen"}:
        return os.path.join(data_root, "train", class_name, object_id, "image")
    if split_name == "gallery_unseen":
        return os.path.join(data_root, "target", object_id, "image")
    if split_name == "query_unseen":
        return os.path.join(data_root, "query", object_id, "image")
    raise ValueError(f"Unsupported split_name: {split_name}")


def resolve_depth_dir(depth_root: str, split_name: str, class_name: str, object_id: str) -> str:
    if split_name in {"train_seen", "val_seen"}:
        return os.path.join(depth_root, "train", class_name, object_id)
    if split_name == "gallery_unseen":
        return os.path.join(depth_root, "target", object_id)
    if split_name == "query_unseen":
        return os.path.join(depth_root, "query", object_id)
    raise ValueError(f"Unsupported split_name: {split_name}")


def encode_rgb_object(encoder: CLIPEncoder, image_dir: str) -> np.ndarray:
    view_paths = list_rgb_views(image_dir)
    if not view_paths:
        raise FileNotFoundError(f"No JPG views found in {image_dir}")
    images = [Image.open(path).convert("RGB") for path in view_paths]
    return encoder.encode_multi_view(images)


def encode_depth_object(encoder: CLIPEncoder, depth_dir: str) -> np.ndarray:
    view_paths = list_depth_views(depth_dir)
    if not view_paths:
        raise FileNotFoundError(f"No depth PNG views found in {depth_dir}")
    images = [load_depth_rgb(path) for path in view_paths]
    return encoder.encode_multi_view(images)


def main():
    args = parse_args()
    output_root = resolve_output_root(args)
    protocol = load_protocol(args.protocol)
    items = build_unique_items(protocol)
    encoder = CLIPEncoder(model_name=args.clip_model, device=args.device or None)

    os.makedirs(output_root, exist_ok=True)
    saved_count = 0
    skipped_count = 0

    print(f"Protocol: {args.protocol}")
    print(f"Modality: {args.modality}")
    print(f"Data root: {args.data_root}")
    if args.modality == "depth":
        print(f"Depth root: {args.depth_root}")
    print(f"Output root: {output_root}")
    print(f"Unique objects: {len(items)}")

    for class_name, item_name, split_name in tqdm(items, desc=f"Extracting {args.modality}"):
        object_id = os.path.splitext(item_name)[0]
        class_output_dir = os.path.join(output_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        output_path = os.path.join(class_output_dir, item_name)

        if os.path.exists(output_path) and not args.overwrite:
            skipped_count += 1
            continue

        if args.modality == "rgb":
            image_dir = resolve_rgb_dir(
                data_root=args.data_root,
                split_name=split_name,
                class_name=class_name,
                object_id=object_id,
            )
            feat = encode_rgb_object(encoder, image_dir=image_dir)
        else:
            depth_dir = resolve_depth_dir(
                depth_root=args.depth_root,
                split_name=split_name,
                class_name=class_name,
                object_id=object_id,
            )
            feat = encode_depth_object(encoder, depth_dir=depth_dir)

        feat = np.asarray(feat, dtype=np.float32)
        np.save(output_path, feat)
        saved_count += 1

    print(f"Saved features: {saved_count}")
    print(f"Skipped existing objects: {skipped_count}")


if __name__ == "__main__":
    main()
