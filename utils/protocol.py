import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np

ClassItems = Dict[str, List[str]]


def build_class_file_dict(feat_root: str) -> Dict[str, List[str]]:
    class_to_files: Dict[str, List[str]] = {}

    for cls in os.listdir(feat_root):
        cls_path = os.path.join(feat_root, cls)
        if not os.path.isdir(cls_path):
            continue

        npy_files = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.endswith(".npy")
        ]
        if npy_files:
            class_to_files[cls] = npy_files

    return class_to_files


def build_common_class_items(rgb_root: str, depth_root: str) -> ClassItems:
    rgb_dict = build_class_file_dict(rgb_root)
    depth_dict = build_class_file_dict(depth_root)

    common: ClassItems = {}
    for cls in sorted(set(rgb_dict.keys()) & set(depth_dict.keys())):
        rgb_names = {os.path.basename(path) for path in rgb_dict[cls]}
        depth_names = {os.path.basename(path) for path in depth_dict[cls]}
        names = sorted(rgb_names & depth_names)
        if len(names) >= 2:
            common[cls] = names

    return common


def _split_items(items: List[str], ratio: float) -> Tuple[List[str], List[str]]:
    if len(items) < 2:
        raise ValueError("Each class must contain at least 2 samples.")

    split_idx = int(len(items) * ratio)
    split_idx = min(max(split_idx, 1), len(items) - 1)
    return items[:split_idx], items[split_idx:]


def build_seen_unseen_protocol(
    class_to_items: ClassItems,
    seen_num: int,
    unseen_num: int,
    seen_train_ratio: float,
    unseen_gallery_ratio: float,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    candidate_classes = sorted(
        cls for cls, items in class_to_items.items() if len(items) >= 2
    )
    if len(candidate_classes) < seen_num + unseen_num:
        raise ValueError(
            f"Not enough classes for protocol: have {len(candidate_classes)}, "
            f"need {seen_num + unseen_num}."
        )

    np.random.shuffle(candidate_classes)
    seen_classes = candidate_classes[:seen_num]
    unseen_classes = candidate_classes[seen_num : seen_num + unseen_num]

    train_seen = {}
    val_seen = {}
    gallery_unseen = {}
    query_unseen = {}

    for cls in seen_classes:
        items = class_to_items[cls].copy()
        random.shuffle(items)
        train_items, val_items = _split_items(items, seen_train_ratio)
        train_seen[cls] = train_items
        val_seen[cls] = val_items

    for cls in unseen_classes:
        items = class_to_items[cls].copy()
        random.shuffle(items)
        gallery_items, query_items = _split_items(items, unseen_gallery_ratio)
        gallery_unseen[cls] = gallery_items
        query_unseen[cls] = query_items

    return {
        "seed": seed,
        "seen_num": seen_num,
        "unseen_num": unseen_num,
        "seen_train_ratio": seen_train_ratio,
        "unseen_gallery_ratio": unseen_gallery_ratio,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
        "train_seen": train_seen,
        "val_seen": val_seen,
        "gallery_unseen": gallery_unseen,
        "query_unseen": query_unseen,
    }


def save_protocol(protocol: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=4)


def load_protocol(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_split_items(protocol: dict, split_name: str) -> List[Tuple[str, str]]:
    split_dict = protocol[split_name]
    items: List[Tuple[str, str]] = []
    for cls in split_dict:
        for item in split_dict[cls]:
            items.append((cls, item))
    return items


def materialize_split_paths(
    protocol: dict,
    split_name: str,
    feat_root: str,
) -> List[Tuple[str, str]]:
    entries = []
    for cls, item in get_split_items(protocol, split_name):
        entries.append((cls, os.path.join(feat_root, cls, item)))
    return entries
