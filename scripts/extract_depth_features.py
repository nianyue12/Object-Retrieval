import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.clip_encoder import CLIPEncoder

# ===== 配置 =====
encoder = CLIPEncoder(model_name="ViT-B/32")
ROOT_IMG_DIR = r"D:/1Ahaha/AA3d/output_224"
OUT_FEAT_DIR = r"D:/1Ahaha/AA3d/output_224_clip_feat_depth"
N_VIEWS = 12

os.makedirs(OUT_FEAT_DIR, exist_ok=True)

def extract_object_feat(obj_dir):
    imgs = []

    for i in range(N_VIEWS):
        img_path = os.path.join(obj_dir, f"depth_{i:04d}.png")
        if not os.path.exists(img_path):
            continue

        pil_img = Image.open(img_path)
        img_array = np.array(pil_img, dtype=np.float32)

        # 8bit 图简单归一化（可有可无，但保持一致）
        if img_array.max() > img_array.min():
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        else:
            img_array = np.zeros_like(img_array)

        img_array = (img_array * 255).astype(np.uint8)

        # 单通道 → 3通道
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        img = Image.fromarray(img_array)
        imgs.append(img)

    if len(imgs) == 0:
        return None

    # ===== CLIP 特征提取 =====
    feat = encoder.encode_multi_view(imgs)

    # 如果是 torch tensor → numpy
    if hasattr(feat, "cpu"):
        feat = feat.cpu().numpy()

    feat = np.array(feat)

    # ===== 自动处理维度 =====
    # 情况1: encode_multi_view 已经返回 (512,)
    if feat.ndim == 1:
        pass

    # 情况2: 返回 (V,512)，需要做平均
    elif feat.ndim == 2:
        feat = feat.mean(axis=0)

    else:
        print("⚠ Unexpected feature shape:", feat.shape)
        return None

    # ===== L2 归一化 =====
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm

    return feat


# ===== 主循环 =====
for cat_dir in sorted(os.listdir(ROOT_IMG_DIR)):
    if not cat_dir.endswith("_multi_view"):
        continue

    cat_name = cat_dir.replace("_multi_view", "")
    full_cat_dir = os.path.join(ROOT_IMG_DIR, cat_dir)

    cat_feat_dir = os.path.join(OUT_FEAT_DIR, cat_name)
    os.makedirs(cat_feat_dir, exist_ok=True)

    print(f"\n📂 Processing category: {cat_name}")

    for obj_id in tqdm(os.listdir(full_cat_dir)):
        obj_dir = os.path.join(full_cat_dir, obj_id)
        if not os.path.isdir(obj_dir):
            continue

        feat = extract_object_feat(obj_dir)
        if feat is None:
            continue

        out_path = os.path.join(cat_feat_dir, f"{obj_id}.npy")
        np.save(out_path, feat)

print("\n✅ All features extracted.")