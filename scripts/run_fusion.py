# scripts/run_fusion.py

import os
import random
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

from utils.metrics import compute_open_set_metrics
from utils.split import build_class_file_dict, build_open_set_split
from utils.features import load_feature
from utils.scoring import score_topk

# ------------------------ 0️⃣ 固定随机种子 ------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# ------------------------ 1️⃣ 路径配置 ------------------------
rgb_root = r"D:/1Ahaha/AA3d/output_224_clip_feat"
depth_root = r"D:/1Ahaha/AA3d/output_224_clip_feat_depth"
save_dir = os.path.join("results", "fusion")
os.makedirs(save_dir, exist_ok=True)

# ------------------------ 2️⃣ 构建公共文件列表 ------------------------
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

# ------------------------ 3️⃣ Open-set split ------------------------
gallery_file_list, query_file_list, query_is_known, KNOWN_CLASSES, UNKNOWN_CLASSES = build_open_set_split(
    class_to_files, known_num=40, unknown_num=10, gallery_ratio=0.7, seed=SEED
)

query_is_known = np.array(query_is_known)

print("Known classes:", KNOWN_CLASSES)
print("Unknown classes:", UNKNOWN_CLASSES)
print("Gallery size:", len(gallery_file_list))
print("Query size:", len(query_file_list))
print("Known queries:", query_is_known.sum(), "Unknown queries:", (query_is_known==0).sum())

# ------------------------ 4️⃣ 特征融合函数 ------------------------
def fuse_feature(rgb_feat, depth_feat, alpha=0.3):
    fused = alpha * rgb_feat + (1 - alpha) * depth_feat
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused

# ------------------------ 5️⃣ 加载融合特征 ------------------------
def load_feats(file_list, alpha=0.3):
    feats = []
    for cls, fname in file_list:
        rgb_feat = load_feature(os.path.join(rgb_root, cls, fname), multi_view=False)
        depth_feat = load_feature(os.path.join(depth_root, cls, fname), multi_view=False)  # ✅ 改成 False
        feats.append(fuse_feature(rgb_feat, depth_feat, alpha))
    return np.stack(feats)

gallery_feats = load_feats(gallery_file_list, alpha=0.3)
query_feats = load_feats(query_file_list, alpha=0.3)

# ------------------------ 6️⃣ 计算相似度 ------------------------
sim_to_gallery = cosine_similarity(query_feats, gallery_feats)

# ------------------------ 7️⃣ 计算评分 ------------------------
scores_max = sim_to_gallery.max(axis=1)
scores_top5 = score_topk(sim_to_gallery, k=5)

# ------------------------ 8️⃣ 计算指标 ------------------------
methods = [("Max", scores_max), ("Top-5 Mean", scores_top5)]
metrics_results = []

for name, scores in methods:
    auroc, fpr95 = compute_open_set_metrics(scores, query_is_known, tpr_target=0.95)
    metrics_results.append({"method": name, "AUROC": float(auroc), "FPR95": float(fpr95)})
    print(f"{name:12} AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")

# ------------------------ 9️⃣ 保存 JSON ------------------------
output = {
    "known_classes": KNOWN_CLASSES,
    "unknown_classes": UNKNOWN_CLASSES,
    "gallery_size": len(gallery_file_list),
    "query_size": len(query_file_list),
    "known_queries": int(query_is_known.sum()),
    "unknown_queries": int((query_is_known==0).sum()),
    "metrics": metrics_results
}

metrics_path = os.path.join(save_dir, "fusion_results.json")
with open(metrics_path, "w") as f:
    json.dump(output, f, indent=4)

print("\nResults saved:", metrics_path)
print("\n✅ Fusion α=0.3 scoring实验完成！")