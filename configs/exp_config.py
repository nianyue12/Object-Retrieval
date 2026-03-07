import os

BASE_DIR = r"D:/1Ahaha/AA3d"

# 特征路径
RGB_FEAT_DIR = os.path.join(BASE_DIR, "output_224_clip_feat")
DEPTH_FEAT_DIR = os.path.join(BASE_DIR, "output_224_clip_feat_depth")

# 保存路径
SAVE_DIR_RGB = os.path.join(BASE_DIR, "3D-Data-Processing-Toolkit", "results", "rgb_mv")
SAVE_DIR_DEPTH = os.path.join(BASE_DIR, "3D-Data-Processing-Toolkit", "results", "depth_mv")
SAVE_DIR_FUSION = os.path.join(BASE_DIR, "3D-Data-Processing-Toolkit", "results", "fusion_mv")

# 实验参数
SEED = 0
GALLERY_RATIO = 0.7
KNOWN_NUM = 40
UNKNOWN_NUM = 10
ALPHA_FUSION = 0.5   # RGB+Depth 融合权重
BATCH_SIZE = 512     # Depth / Fusion 分批计算相似度防止爆内存


# 加上这一行
RESULT_DIR = os.path.join(BASE_DIR, "3D-Data-Processing-Toolkit", "results")