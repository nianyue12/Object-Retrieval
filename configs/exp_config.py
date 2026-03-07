import os

BASE_DIR = r"D:/1Ahaha/AA3d"
PROJECT_DIR = os.path.join(BASE_DIR, "3D-Data-Processing-Toolkit")

# 特征路径
RGB_FEAT_DIR = os.path.join(BASE_DIR, "output_224_clip_feat")
DEPTH_FEAT_DIR = os.path.join(BASE_DIR, "output_224_clip_feat_depth")

# 保存路径
RESULT_DIR = os.path.join(PROJECT_DIR, "results")

RGB_RESULT_DIR = os.path.join(RESULT_DIR, "rgb")
DEPTH_RESULT_DIR = os.path.join(RESULT_DIR, "depth")
FUSION_RESULT_DIR = os.path.join(RESULT_DIR, "fusion")

# 实验参数
SEED = 0
GALLERY_RATIO = 0.7
KNOWN_NUM = 40
UNKNOWN_NUM = 10
ALPHA_FUSION = 0.3
BATCH_SIZE = 512        # Depth / Fusion 分批计算相似度防止爆内存