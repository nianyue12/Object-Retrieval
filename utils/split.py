# utils/split.py

import os
import random
import numpy as np


def build_class_file_dict(feat_root):
    """
    构建:
    {
        class_name: [file1.npy, file2.npy, ...]
    }
    """
    class_to_files = {}

    for cls in os.listdir(feat_root):
        cls_path = os.path.join(feat_root, cls)
        if not os.path.isdir(cls_path):
            continue

        npy_files = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.endswith(".npy")
        ]

        if len(npy_files) > 0:
            class_to_files[cls] = npy_files

    return class_to_files


def build_open_set_split(
    class_to_files,
    known_num=40,
    unknown_num=10,
    gallery_ratio=0.7,
    seed=0,
):
    """
    返回:
        gallery_files  : list[(cls, file_path)]
        query_files    : list[(cls, file_path)]
        query_is_known : list[int]
        known_classes
        unknown_classes
    """

    random.seed(seed)
    np.random.seed(seed)

    all_classes = sorted(class_to_files.keys())

    if len(all_classes) < known_num + unknown_num:
        raise ValueError("类别数量不足")

    np.random.shuffle(all_classes)

    known_classes = all_classes[:known_num]
    unknown_classes = all_classes[known_num : known_num + unknown_num]

    gallery_files = []
    query_files = []
    query_is_known = []

    # -------- 已知类 --------
    for cls in known_classes:
        files = class_to_files[cls].copy()
        random.shuffle(files)

        split_idx = int(len(files) * gallery_ratio)
        gallery_part = files[:split_idx]
        query_part = files[split_idx:]

        for f in gallery_part:
            gallery_files.append((cls, f))

        for f in query_part:
            query_files.append((cls, f))
            query_is_known.append(1)

    # -------- 未知类（全部进 query）--------
    for cls in unknown_classes:
        files = class_to_files[cls]
        for f in files:
            query_files.append((cls, f))
            query_is_known.append(0)

    return (
        gallery_files,
        query_files,
        query_is_known,
        known_classes,
        unknown_classes,
    )