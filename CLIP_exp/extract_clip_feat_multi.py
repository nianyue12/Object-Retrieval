import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

# ===== 配置 =====
ROOT_IMG_DIR = r"D:/1Ahaha/AA3d/output_224"
OUT_FEAT_DIR = r"D:/1Ahaha/AA3d/output_224_clip_feat"
N_VIEWS = 12

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

os.makedirs(OUT_FEAT_DIR, exist_ok=True)

def extract_object_feat(obj_dir):
    feats = []
    for i in range(N_VIEWS):
        img_path = os.path.join(obj_dir, f"rgb_{i:04d}.png")
        if not os.path.exists(img_path):
            continue
        img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.cpu().numpy())
    if len(feats) == 0:
        return None
    feats = np.concatenate(feats, axis=0)   # [V, 512]
    return feats.mean(axis=0)                # [512]

# ===== 主循环 =====
for cat_dir in sorted(os.listdir(ROOT_IMG_DIR)):
    if not cat_dir.endswith("_multi_view"):
        continue

    cat_name = cat_dir.replace("_multi_view", "")
    full_cat_dir = os.path.join(ROOT_IMG_DIR, cat_dir)

    print(f"\n📂 Processing category: {cat_name}")

    for obj_id in tqdm(os.listdir(full_cat_dir)):
        obj_dir = os.path.join(full_cat_dir, obj_id)
        if not os.path.isdir(obj_dir):
            continue

        feat = extract_object_feat(obj_dir)
        if feat is None:
            continue

        out_name = f"{cat_name}_{obj_id}.npy"
        out_path = os.path.join(OUT_FEAT_DIR, out_name)
        np.save(out_path, feat)

print("\n✅ All features extracted.")
