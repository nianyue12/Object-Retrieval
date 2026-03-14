import os

BASE_DIR = r"D:/1Ahaha/AA3d"
PROJECT_DIR = os.path.join(BASE_DIR, "3D-Data-Processing-Toolkit")

# Feature roots
RGB_FEAT_DIR = os.path.join(BASE_DIR, "output_224_clip_feat")
DEPTH_FEAT_DIR = os.path.join(BASE_DIR, "output_feat_depth_maps")

# Result and split paths
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
SPLIT_DIR = os.path.join(PROJECT_DIR, "configs", "splits")

RGB_RESULT_DIR = os.path.join(RESULT_DIR, "rgb")
DEPTH_RESULT_DIR = os.path.join(RESULT_DIR, "depth")
FUSION_RESULT_DIR = os.path.join(RESULT_DIR, "fusion")
UNSEEN_RESULT_DIR = os.path.join(RESULT_DIR, "unseen_retrieval")
SEMANTIC_RESULT_DIR = os.path.join(RESULT_DIR, "semantic_training")

# Experiment parameters
SEED = 0
GALLERY_RATIO = 0.7
KNOWN_NUM = 10
UNKNOWN_NUM = 40
ALPHA_FUSION = 0.5
SEEN_TRAIN_RATIO = 0.8
UNSEEN_GALLERY_RATIO = 0.7
BATCH_SIZE = 512
TRAIN_BATCH_SIZE = 256
TRAIN_EPOCHS = 20
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TEXT_LOSS_WEIGHT = 1.0
MODAL_LOSS_WEIGHT = 0.1
TEXT_TEMPERATURE = 0.07
PROJ_DIM = 512

DEFAULT_PROTOCOL_PATH = os.path.join(
    SPLIT_DIR,
    f"shapenet_seen{KNOWN_NUM}_unseen{UNKNOWN_NUM}_seed{SEED}.json",
)
