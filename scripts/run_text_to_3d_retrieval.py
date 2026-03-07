import os
import numpy as np
import torch
import clip

# =============================
# 配置
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 你的RGB特征目录
GALLERY_FEAT_DIR = r"D:/1Ahaha/AA3d/output_224_clip_feat"

TOPK = 10

TEXT_PROMPTS = [
    "a 3D model of a chair",
    "a 3D model of a car",
    "a 3D model of a guitar",
    "a 3D model of a airplane",
]

# =============================
# 加载CLIP
# =============================

print("Loading CLIP model...")

model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# =============================
# 加载3D特征（递归读取）
# =============================

print("Loading 3D features...")

obj_names = []
gallery_feats = []

for root, dirs, files in os.walk(GALLERY_FEAT_DIR):

    for file in files:

        if not file.endswith(".npy"):
            continue

        path = os.path.join(root, file)

        feat = np.load(path)

        # 相对路径
        rel_path = os.path.relpath(path, GALLERY_FEAT_DIR)

        # 转成统一格式
        obj_id = rel_path.replace("\\", "/").replace(".npy", "")

        gallery_feats.append(feat)
        obj_names.append(obj_id)

gallery_feats = np.stack(gallery_feats)

print("Loaded objects:", len(obj_names))
print("Feature shape:", gallery_feats.shape)

# =============================
# 特征归一化
# =============================

gallery_feats = gallery_feats / np.linalg.norm(
    gallery_feats, axis=1, keepdims=True
)

# =============================
# Text → 3D Retrieval
# =============================

for prompt in TEXT_PROMPTS:

    print("\n==============================")
    print("Text query:", prompt)

    text = clip.tokenize([prompt]).to(DEVICE)

    with torch.no_grad():
        text_feat = model.encode_text(text)

    text_feat = text_feat.cpu().numpy()[0]

    # 归一化
    text_feat = text_feat / np.linalg.norm(text_feat)

    # cosine similarity
    sims = gallery_feats @ text_feat

    # TopK
    topk_idx = np.argsort(-sims)[:TOPK]

    print("\nTop", TOPK, "results:")

    for rank, idx in enumerate(topk_idx):

        name = obj_names[idx]
        score = sims[idx]

        print(f"{rank+1:02d}. {name}  score={score:.4f}")