import os

BASE_DIR = r"D:/1Ahaha/AA3d"
PROJECT_DIR = os.path.join(BASE_DIR, "3D-Data-Processing-Toolkit")

# Feature roots
RGB_FEAT_DIR = os.path.join(BASE_DIR, "output_224_clip_feat")
DEPTH_FEAT_DIR = os.path.join(BASE_DIR, "output_feat_depth_maps")

# Output and split paths
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
SPLIT_DIR = os.path.join(PROJECT_DIR, "configs", "splits")
UNSEEN_RESULT_DIR = os.path.join(RESULT_DIR, "unseen_retrieval")

# Protocol settings
SEED = 0
KNOWN_NUM = 10
UNKNOWN_NUM = 40
SEEN_TRAIN_RATIO = 0.8
UNSEEN_GALLERY_RATIO = 0.7

# Retrieval defaults
ALPHA_FUSION = 0.5
BATCH_SIZE = 512

DEFAULT_PROTOCOL_PATH = os.path.join(
    SPLIT_DIR,
    f"shapenet_seen{KNOWN_NUM}_unseen{UNKNOWN_NUM}_seed{SEED}.json",
)
