"""
功能：批量提取深度图的 CLIP 多视图特征。

说明：
    这个脚本会遍历每个物体的 12 张深度图，
    把深度图转成 3 通道图像后送入 CLIP 编码，
    最终为每个物体保存一个 `.npy` 融合特征。
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
# 深度图也复用 CLIPEncoder，区别在于送入前会先转成 3 通道图像。
encoder = CLIPEncoder(model_name="ViT-B/32")
ROOT_IMG_DIR = r"D:/1Ahaha/AA3d/depth_maps"
OUT_FEAT_DIR = r"D:/1Ahaha/AA3d/output_feat_depth_maps"
N_VIEWS = 12

os.makedirs(OUT_FEAT_DIR, exist_ok=True)

def extract_object_feat(obj_dir):
    """
    功能：提取单个物体的深度多视图特征。
    """
    imgs = []

    for i in range(N_VIEWS):
        img_path = os.path.join(obj_dir, f"depth_{i:02d}.png")  # depth_00.png
        # img_path = os.path.join(obj_dir, f"depth_{i:04d}.png")
        if not os.path.exists(img_path):
            continue

        pil_img = Image.open(img_path)
        img_array = np.array(pil_img, dtype=np.float32)

        # 8bit 图简单归一化（可有可无，但保持一致）
        if img_array.max() > img_array.min():
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        else:
            img_array = np.zeros_like(img_array)

        # CLIP 预处理期望图像输入，这里把归一化深度重新映射到 8bit。
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
# 遍历每个类别目录，并为其中的每个物体提取特征
for cat_dir in sorted(os.listdir(ROOT_IMG_DIR)):
    full_cat_dir = os.path.join(ROOT_IMG_DIR, cat_dir)
    if not os.path.isdir(full_cat_dir):
        continue

    # 深度图目录已经按类别命名，不需要像 RGB 那样去掉 `_multi_view` 后缀。
    cat_name = cat_dir
    cat_feat_dir = os.path.join(OUT_FEAT_DIR, cat_name)
    os.makedirs(cat_feat_dir, exist_ok=True)

    print(f"\n📂 Processing category: {cat_name}")

# for cat_dir in sorted(os.listdir(ROOT_IMG_DIR)):
#     if not cat_dir.endswith("_multi_view"):
#         continue

#     cat_name = cat_dir.replace("_multi_view", "")
#     full_cat_dir = os.path.join(ROOT_IMG_DIR, cat_dir)

#     cat_feat_dir = os.path.join(OUT_FEAT_DIR, cat_name)
#     os.makedirs(cat_feat_dir, exist_ok=True)

#     print(f"\n📂 Processing category: {cat_name}")


    for obj_id in tqdm(os.listdir(full_cat_dir)):
        obj_dir = os.path.join(full_cat_dir, obj_id)
        if not os.path.isdir(obj_dir):
            continue

        feat = extract_object_feat(obj_dir)
        if feat is None:
            continue

        out_path = os.path.join(cat_feat_dir, f"{obj_id}.npy")
        # 输出路径与 RGB 特征保持同样的 `<class>/<object>.npy` 结构。
        np.save(out_path, feat)

print("\n✅ All features extracted.")
