"""
功能：根据已有的 RGB / Depth 特征目录，生成 seen / unseen 协议文件。
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import (
    DEFAULT_PROTOCOL_PATH,
    DEPTH_FEAT_DIR,
    KNOWN_NUM,
    RGB_FEAT_DIR,
    SEED,
    SEEN_TRAIN_RATIO,
    UNKNOWN_NUM,
    UNSEEN_GALLERY_RATIO,
)
from utils.protocol import (
    build_common_class_items,
    build_seen_unseen_protocol,
    save_protocol,
)


def main():
    """脚本入口：构建协议并输出基础统计信息。"""
    # 先找出 RGB 和深度两边都存在的公共样本
    class_to_items = build_common_class_items(RGB_FEAT_DIR, DEPTH_FEAT_DIR)
    protocol = build_seen_unseen_protocol(
        class_to_items=class_to_items,
        seen_num=KNOWN_NUM,
        unseen_num=UNKNOWN_NUM,
        seen_train_ratio=SEEN_TRAIN_RATIO,
        unseen_gallery_ratio=UNSEEN_GALLERY_RATIO,
        seed=SEED,
    )

    # 保存协议文件，供训练和检索脚本复用
    save_protocol(protocol, DEFAULT_PROTOCOL_PATH)

    gallery_size = sum(len(v) for v in protocol["gallery_unseen"].values())
    query_size = sum(len(v) for v in protocol["query_unseen"].values())
    train_size = sum(len(v) for v in protocol["train_seen"].values())
    val_size = sum(len(v) for v in protocol["val_seen"].values())

    print(f"Saved protocol: {DEFAULT_PROTOCOL_PATH}")
    print(f"Seen classes ({len(protocol['seen_classes'])}): {protocol['seen_classes']}")
    print(f"Unseen classes ({len(protocol['unseen_classes'])}): {protocol['unseen_classes']}")
    print(f"Train seen samples: {train_size}")
    print(f"Val seen samples: {val_size}")
    print(f"Gallery unseen samples: {gallery_size}")
    print(f"Query unseen samples: {query_size}")


if __name__ == "__main__":
    main()
