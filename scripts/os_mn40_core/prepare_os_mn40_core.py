"""
功能：为 OS-MN40-core 数据集生成项目可用的协议文件。
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import BASE_DIR

DATA_ROOT = os.path.join(BASE_DIR, "OS_MN40_core")
RAW_ROOT = os.path.join(BASE_DIR, "OS_MN40_core_raw")
# 默认协议文件会被后续 OS-MN40-core 特征提取和检索脚本复用。
DEFAULT_PROTOCOL_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "splits",
    "os_mn40_core_seen8_unseen32_seed0.json",
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Prepare an OS-MN40-core protocol file for this project."
    )
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument(
        "--query_label",
        type=str,
        default=os.path.join(RAW_ROOT, "query_label.txt"),
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=os.path.join(RAW_ROOT, "target_label.txt"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seen_train_ratio", type=float, default=0.8)
    parser.add_argument("--output_path", type=str, default=DEFAULT_PROTOCOL_PATH)
    return parser.parse_args()


def _split_items(items: List[str], ratio: float) -> Tuple[List[str], List[str]]:
    """按比例把 seen 类样本切成训练集和验证集。"""
    if len(items) < 2:
        raise ValueError("Each seen class must contain at least 2 samples.")

    split_idx = int(len(items) * ratio)
    split_idx = min(max(split_idx, 1), len(items) - 1)
    return items[:split_idx], items[split_idx:]


def read_label_file(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    功能：读取 `object_id,class_name` 格式的标签文件。
    """
    class_to_items: Dict[str, List[str]] = defaultdict(list)
    item_to_class: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # 原始标签文件每行格式为 object_id,class_name。
            parts = line.split(",", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid label line at {path}:{line_no}: {line}")

            object_id, class_name = parts[0].strip(), parts[1].strip()
            item_name = f"{object_id}.npy"
            class_to_items[class_name].append(item_name)
            item_to_class[item_name] = class_name

    for class_name in class_to_items:
        class_to_items[class_name] = sorted(class_to_items[class_name])

    return dict(class_to_items), item_to_class


def discover_seen_items(train_root: str) -> Dict[str, List[str]]:
    """
    功能：扫描 train 目录，收集 seen 类样本。
    """
    seen_dict: Dict[str, List[str]] = {}

    for class_name in sorted(os.listdir(train_root)):
        # train 目录按类别组织，因此这里扫描出的类别就是 seen 类。
        class_dir = os.path.join(train_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        object_items = []
        for object_id in sorted(os.listdir(class_dir)):
            object_dir = os.path.join(class_dir, object_id)
            if not os.path.isdir(object_dir):
                continue
            object_items.append(f"{object_id}.npy")

        if object_items:
            seen_dict[class_name] = object_items

    return seen_dict


def validate_source_dirs(
    data_root: str,
    seen_items: Dict[str, List[str]],
    query_labels: Dict[str, List[str]],
    target_labels: Dict[str, List[str]],
) -> None:
    """
    功能：检查协议里引用到的源目录是否都存在。
    """
    train_root = os.path.join(data_root, "train")
    query_root = os.path.join(data_root, "query")
    target_root = os.path.join(data_root, "target")

    # 协议中写入的每个 item 都要能在源数据目录中找到对应物体目录。
    for class_name, items in seen_items.items():
        for item in items:
            object_id = os.path.splitext(item)[0]
            object_dir = os.path.join(train_root, class_name, object_id)
            if not os.path.isdir(object_dir):
                raise FileNotFoundError(f"Missing seen object directory: {object_dir}")

    for class_name, items in query_labels.items():
        for item in items:
            object_id = os.path.splitext(item)[0]
            object_dir = os.path.join(query_root, object_id)
            if not os.path.isdir(object_dir):
                raise FileNotFoundError(
                    f"Missing query object directory for class {class_name}: {object_dir}"
                )

    for class_name, items in target_labels.items():
        for item in items:
            object_id = os.path.splitext(item)[0]
            object_dir = os.path.join(target_root, object_id)
            if not os.path.isdir(object_dir):
                raise FileNotFoundError(
                    f"Missing target object directory for class {class_name}: {object_dir}"
                )


def main():
    """脚本入口：读取源标签、切分 seen 集、保存协议。"""
    args = parse_args()
    random.seed(args.seed)

    train_root = os.path.join(args.data_root, "train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Missing train root: {train_root}")

    seen_items = discover_seen_items(train_root)
    query_labels, _ = read_label_file(args.query_label)
    target_labels, _ = read_label_file(args.target_label)
    validate_source_dirs(args.data_root, seen_items, query_labels, target_labels)

    seen_classes = sorted(seen_items.keys())
    unseen_classes = sorted(set(query_labels.keys()) | set(target_labels.keys()))
    overlap = sorted(set(seen_classes) & set(unseen_classes))
    # OS-MN40-core 的开放集设置要求 seen 类和 unseen 类互不重叠。
    if overlap:
        raise ValueError(
            f"OS-MN40-core seen/unseen classes should not overlap, but found {overlap}"
        )

    # seen 类进一步切成训练和验证，unseen 类直接沿用官方 query / target
    train_seen: Dict[str, List[str]] = {}
    val_seen: Dict[str, List[str]] = {}
    for class_name in seen_classes:
        items = seen_items[class_name].copy()
        random.shuffle(items)
        train_items, val_items = _split_items(items, args.seen_train_ratio)
        train_seen[class_name] = sorted(train_items)
        val_seen[class_name] = sorted(val_items)

    protocol = {
        "dataset": "os_mn40_core",
        "data_root": args.data_root,
        "query_label_path": args.query_label,
        "target_label_path": args.target_label,
        "seed": args.seed,
        "seen_num": len(seen_classes),
        "unseen_num": len(unseen_classes),
        "seen_train_ratio": args.seen_train_ratio,
        "view_count": 24,
        "point_count": 1024,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
        "train_seen": train_seen,
        "val_seen": val_seen,
        "gallery_unseen": target_labels,
        "query_unseen": query_labels,
        "source_layout": {
            "train_seen": "data_root/train/<class_name>/<object_id>",
            "val_seen": "data_root/train/<class_name>/<object_id>",
            "gallery_unseen": "data_root/target/<object_id>",
            "query_unseen": "data_root/query/<object_id>",
        },
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=4)

    train_count = sum(len(v) for v in train_seen.values())
    val_count = sum(len(v) for v in val_seen.values())
    query_count = sum(len(v) for v in query_labels.values())
    gallery_count = sum(len(v) for v in target_labels.values())

    print(f"Saved protocol: {args.output_path}")
    print(f"Seen classes ({len(seen_classes)}): {seen_classes}")
    print(f"Unseen classes ({len(unseen_classes)}): {unseen_classes}")
    print(f"Train seen samples: {train_count}")
    print(f"Val seen samples: {val_count}")
    print(f"Gallery unseen samples: {gallery_count}")
    print(f"Query unseen samples: {query_count}")


if __name__ == "__main__":
    main()
