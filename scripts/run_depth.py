# scripts/run_depth.py

import os
import random
import numpy as np
import json
from tqdm import tqdm

from utils.metrics import compute_map, compute_ndcg  
from utils.split import build_class_file_dict, build_open_set_split
from utils.features import load_feature

from configs.exp_config import DEPTH_FEAT_DIR, RESULT_DIR, KNOWN_NUM, UNKNOWN_NUM, GALLERY_RATIO, SEED, BATCH_SIZE

# ------------------------ 固定随机种子 ------------------------
random.seed(SEED)
np.random.seed(SEED)

# ------------------------ 结果目录 ------------------------
save_dir = os.path.join(RESULT_DIR, "depth")
os.makedirs(save_dir, exist_ok=True)

# ------------------------ 构建类别 → 文件列表 ------------------------
class_to_files = build_class_file_dict(DEPTH_FEAT_DIR)

# ------------------------ Open-set split ------------------------
gallery_file_list, query_file_list, query_is_known, KNOWN_CLASSES, UNKNOWN_CLASSES = build_open_set_split(
    class_to_files,
    known_num=KNOWN_NUM,
    unknown_num=UNKNOWN_NUM,
    gallery_ratio=GALLERY_RATIO,
    seed=SEED
)

print("Known classes:", KNOWN_CLASSES)
print("Unknown classes:", UNKNOWN_CLASSES)

# ------------------------ 加载特征 ------------------------
gallery_feats = [load_feature(f, multi_view=False) for cls, f in gallery_file_list]
query_feats = [load_feature(f, multi_view=False) for cls, f in query_file_list]
query_is_known = np.array(query_is_known)

gallery_feats = np.stack(gallery_feats)
query_feats = np.stack(query_feats)

# ------------------------ 过滤 unknown queries ------------------------
known_mask = query_is_known == 1
query_feats = query_feats[known_mask]
query_labels = [cls for i, (cls, f) in enumerate(query_file_list) if known_mask[i]]
gallery_labels = [cls for cls, f in gallery_file_list]

print("Gallery size:", gallery_feats.shape)
print("Query size:", query_feats.shape)

# ------------------------ L2归一化 ------------------------
gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)
query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)

gallery_labels = np.array(gallery_labels)
query_labels = np.array(query_labels)

# ------------------------ 计算相似度矩阵 ------------------------
print("\n计算相似度矩阵...")
sim_to_gallery = np.zeros((query_feats.shape[0], gallery_feats.shape[0]), dtype=np.float32)
for start in tqdm(range(0, query_feats.shape[0], BATCH_SIZE), desc="Computing similarity"):
    end = min(start + BATCH_SIZE, query_feats.shape[0])
    sim_to_gallery[start:end] = query_feats[start:end] @ gallery_feats.T

# ------------------------ 计算 Retrieval 指标 ------------------------
print("\n计算 Retrieval Metrics...")

mAP = compute_map(sim_to_gallery, gallery_labels, query_labels)
ndcg = compute_ndcg(sim_to_gallery, gallery_labels, query_labels)

metrics_results = {
    "mAP": float(mAP),
    "NDCG": float(ndcg)
}

print("\nDepth Retrieval Results:")
print(f"mAP  : {mAP:.4f}")
print(f"NDCG : {ndcg:.4f}")

# ------------------------ 保存 JSON ------------------------
output = {
    "known_classes": KNOWN_CLASSES,
    "unknown_classes": UNKNOWN_CLASSES,
    "gallery_size": int(gallery_feats.shape[0]),
    "query_size": int(query_feats.shape[0]),
    "metrics": metrics_results
}

metrics_path = os.path.join(save_dir, "depth_results.json")
with open(metrics_path, "w") as f:
    json.dump(output, f, indent=4)

print("\nResults saved:", metrics_path)
print("\n✅ Depth scoring实验完成！")

