import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from CLIP_exp.models.clip_encoder import CLIPEncoder

# ===== 配置 =====
encoder = CLIPEncoder(model_name="ViT-B/32")
ROOT_IMG_DIR = r"D:/1Ahaha/AA3d/output_224"
OUT_FEAT_DIR = r"D:/1Ahaha/AA3d/output_224_clip_feat"
N_VIEWS = 12


os.makedirs(OUT_FEAT_DIR, exist_ok=True)

def extract_object_feat(obj_dir):
    imgs = []
    for i in range(N_VIEWS):
        img_path = os.path.join(obj_dir, f"rgb_{i:04d}.png")
        if not os.path.exists(img_path):
            continue
        imgs.append(Image.open(img_path).convert("RGB"))

    if len(imgs) == 0:
        return None
    
    feat = encoder.encode_multi_view(imgs)   # [V, 512]
    return feat                # [512]

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
