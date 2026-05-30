"""
功能：批量提取 RGB 多视图图像的 CLIP 特征。
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
from tqdm import tqdm
from models.clip_encoder import CLIPEncoder

# ===== 配置 =====
# 这里直接实例化 CLIPEncoder，后续每个物体目录复用同一个 CLIP 模型。
encoder = CLIPEncoder(model_name="ViT-B/32")
ROOT_IMG_DIR = r"D:/1Ahaha/AA3d/output_224"
OUT_FEAT_DIR = r"D:/1Ahaha/AA3d/output_224_clip_feat"
N_VIEWS = 12


os.makedirs(OUT_FEAT_DIR, exist_ok=True)

def extract_object_feat(obj_dir):
    """
    功能：提取单个物体的 RGB 多视图特征。
    """
    imgs = []
    for i in range(N_VIEWS):
        img_path = os.path.join(obj_dir, f"rgb_{i:04d}.png")
        if not os.path.exists(img_path):
            # 某些物体可能缺少个别视图，直接跳过缺失图片。
            continue
        imgs.append(Image.open(img_path).convert("RGB"))

    if len(imgs) == 0:
        return None

    # CLIPEncoder 内部会对多视图特征做平均池化并归一化
    feat = encoder.encode_multi_view(imgs)
    return feat

# ===== 主循环 =====
for cat_dir in sorted(os.listdir(ROOT_IMG_DIR)):
    if not cat_dir.endswith("_multi_view"):
        continue

    # 输入目录名是 `<class>_multi_view`，输出特征目录只保留类别名。
    cat_name = cat_dir.replace("_multi_view", "")
    full_cat_dir = os.path.join(ROOT_IMG_DIR, cat_dir)

    # 在类别循环开始时就创建目录（只需创建一次）
    cat_feat_dir = os.path.join(OUT_FEAT_DIR, cat_name)
    os.makedirs(cat_feat_dir, exist_ok=True)

    print(f"\n📂 Processing category: {cat_name}")

    # 遍历当前类别下的每个物体目录
    for obj_id in tqdm(os.listdir(full_cat_dir)):
        obj_dir = os.path.join(full_cat_dir, obj_id)
        if not os.path.isdir(obj_dir):
            continue

        feat = extract_object_feat(obj_dir)
        if feat is None:
            continue

        out_path = os.path.join(cat_feat_dir, f"{obj_id}.npy")
        # 每个物体保存一个融合后的 CLIP 特征。
        np.save(out_path, feat)

print("\n✅ All features extracted.")

