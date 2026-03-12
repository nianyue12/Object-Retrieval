# scripts/run_fusion.py
import os
import random
import numpy as np
import json
from tqdm import tqdm

from utils.metrics import compute_map, compute_ndcg
from utils.split import build_class_file_dict, build_open_set_split
from utils.features import load_feature
from configs.exp_config import BATCH_SIZE

# ------------------------ 固定随机种子 ------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# ------------------------ 路径配置 ------------------------
rgb_root = r"D:/1Ahaha/AA3d/output_224_clip_feat"
depth_root = r"D:/1Ahaha/AA3d/output_feat_depth_maps"
save_dir = os.path.join("results", "fusion")
os.makedirs(save_dir, exist_ok=True)

# ------------------------ 构建公共文件列表 ------------------------
def build_common_class_files(rgb_root, depth_root):
    rgb_dict = {cls: [os.path.basename(f) for f in files] 
                for cls, files in build_class_file_dict(rgb_root).items()}
    depth_dict = {cls: [os.path.basename(f) for f in files] 
                  for cls, files in build_class_file_dict(depth_root).items()}

    common_classes = {}
    for cls in rgb_dict:
        if cls in depth_dict:
            common_files = sorted(list(set(rgb_dict[cls]) & set(depth_dict[cls])))
            if len(common_files) > 0:
                common_classes[cls] = common_files
    return common_classes

class_to_files = build_common_class_files(rgb_root, depth_root)

# ------------------------ Open-set split ------------------------
gallery_file_list, query_file_list, query_is_known, KNOWN_CLASSES, UNKNOWN_CLASSES = build_open_set_split(
    class_to_files, known_num=40, unknown_num=10, gallery_ratio=0.7, seed=SEED
)
query_is_known = np.array(query_is_known)

print("Known classes:", KNOWN_CLASSES)
print("Unknown classes:", UNKNOWN_CLASSES)
print("Gallery size:", len(gallery_file_list))
print("Query size:", len(query_file_list))

# ------------------------ 特征融合函数 ------------------------
def fuse_feature(rgb_feat, depth_feat, alpha=0.5):
    fused = alpha * rgb_feat + (1 - alpha) * depth_feat
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused

def load_feats(file_list, alpha=0.5):
    feats = []
    labels = []
    for cls, fname in file_list:
        rgb_feat = load_feature(os.path.join(rgb_root, cls, fname), multi_view=False)
        depth_feat = load_feature(os.path.join(depth_root, cls, fname), multi_view=False)
        feats.append(fuse_feature(rgb_feat, depth_feat, alpha))
        labels.append(cls)
    return np.stack(feats), np.array(labels)

gallery_feats, gallery_labels = load_feats(gallery_file_list, alpha=0.5)
query_feats, query_labels = load_feats(query_file_list, alpha=0.5)

# ------------------------ 过滤 unknown queries ------------------------
query_is_known = np.array(query_is_known)
known_mask = query_is_known == 1
query_feats = query_feats[known_mask]
query_labels = [query_labels[i] for i in range(len(query_labels)) if known_mask[i]]

print("Known queries used for retrieval:", len(query_labels))

# ------------------------ 计算相似度矩阵 ------------------------
print("\n计算相似度矩阵...")
sim_to_gallery = np.zeros((query_feats.shape[0], gallery_feats.shape[0]), dtype=np.float32)
for start in tqdm(range(0, query_feats.shape[0], BATCH_SIZE), desc="Computing similarity"):
    end = min(start + BATCH_SIZE, query_feats.shape[0])
    sim_to_gallery[start:end] = query_feats[start:end] @ gallery_feats.T

# ------------------------ 计算 Retrieval 指标 ------------------------
mAP = compute_map(sim_to_gallery, gallery_labels, query_labels)
ndcg = compute_ndcg(sim_to_gallery, gallery_labels, query_labels)

metrics_results = {"mAP": float(mAP), "NDCG": float(ndcg)}

print("\nFusion α=0.5 Retrieval Results:")
print(f"mAP  : {mAP:.4f}")
print(f"NDCG : {ndcg:.4f}")

# ------------------------ 保存 JSON ------------------------
output = {
    "known_classes": KNOWN_CLASSES,
    "unknown_classes": UNKNOWN_CLASSES,
    "gallery_size": len(gallery_file_list),
    "query_size": len(query_file_list),
    "metrics": metrics_results
}

metrics_path = os.path.join(save_dir, "fusion_results.json")
with open(metrics_path, "w") as f:
    json.dump(output, f, indent=4)

print("\nResults saved:", metrics_path)
print("\n✅ Fusion α=0.5 scoring实验完成！")


