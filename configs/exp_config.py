"""
功能：集中管理项目默认实验配置。

说明：
    这里主要保存数据根目录、特征目录、协议划分参数，
    以及检索脚本默认会读取的路径和超参数。
"""

import os

# 项目根目录
BASE_DIR = r"D:/1Ahaha/AA3d"
PROJECT_DIR = os.path.join(BASE_DIR, "3D-Data-Processing-Toolkit")

# 特征文件根目录
# RGB_FEAT_DIR 和 DEPTH_FEAT_DIR 是大多数检索/训练脚本默认读取的缓存特征位置。
RGB_FEAT_DIR = os.path.join(BASE_DIR, "output_224_clip_feat")
DEPTH_FEAT_DIR = os.path.join(BASE_DIR, "output_feat_depth_maps")

# 结果目录与协议文件目录
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
SPLIT_DIR = os.path.join(PROJECT_DIR, "configs", "splits")
UNSEEN_RESULT_DIR = os.path.join(RESULT_DIR, "unseen_retrieval")

# Seen / unseen 协议默认参数
# KNOWN_NUM/UNKNOWN_NUM 控制基础 ShapeNet 协议中 seen 与 unseen 类别数量。
SEED = 0
KNOWN_NUM = 10
UNKNOWN_NUM = 40
SEEN_TRAIN_RATIO = 0.8
UNSEEN_GALLERY_RATIO = 0.7

# 检索阶段默认超参数
# ALPHA_FUSION 表示 RGB 权重，Depth 权重对应 1 - ALPHA_FUSION。
ALPHA_FUSION = 0.5
BATCH_SIZE = 512

# 默认协议文件路径
DEFAULT_PROTOCOL_PATH = os.path.join(
    SPLIT_DIR,
    f"shapenet_seen{KNOWN_NUM}_unseen{UNKNOWN_NUM}_seed{SEED}.json",
)
