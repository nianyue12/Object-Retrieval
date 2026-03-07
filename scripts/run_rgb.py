# scripts/run_rgb.py

import os
import random
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

from utils.metrics import compute_open_set_metrics
from utils.split import build_class_file_dict, build_open_set_split
from utils.features import load_feature
from configs.exp_config import RGB_FEAT_DIR, RESULT_DIR, KNOWN_NUM, UNKNOWN_NUM, GALLERY_RATIO, SEED
from utils.scoring import score_topk

# ------------------------ 0️⃣ 固定随机种子 ------------------------
random.seed(SEED)
np.random.seed(SEED)

# ------------------------ 1️⃣ 结果目录 ------------------------
save_dir = os.path.join(RESULT_DIR, "rgb")
os.makedirs(save_dir, exist_ok=True)

# ------------------------ 2️⃣ 构建类别 → 文件列表 ------------------------
class_to_files = build_class_file_dict(RGB_FEAT_DIR)

# ------------------------ 3️⃣ Open-set split ------------------------
gallery_file_list, query_file_list, query_is_known, KNOWN_CLASSES, UNKNOWN_CLASSES = build_open_set_split(
    class_to_files,
    known_num=KNOWN_NUM,
    unknown_num=UNKNOWN_NUM,
    gallery_ratio=GALLERY_RATIO,
    seed=SEED
)

print("Known classes:", KNOWN_CLASSES)
print("Unknown classes:", UNKNOWN_CLASSES)

# ------------------------ 4️⃣ 加载特征 ------------------------
gallery_feats = [load_feature(f, multi_view=False) for cls, f in gallery_file_list]
query_feats = [load_feature(f, multi_view=False) for cls, f in query_file_list]
query_is_known = np.array(query_is_known)

gallery_feats = np.stack(gallery_feats)
query_feats = np.stack(query_feats)

# ------------------------ 5️⃣ L2归一化 ------------------------
gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)
query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)

print("Gallery size:", gallery_feats.shape)
print("Query size:", query_feats.shape)
print("Known queries:", query_is_known.sum(), "Unknown queries:", (query_is_known==0).sum())

# ------------------------ 6️⃣ 计算相似度矩阵 ------------------------
print("\n计算相似度矩阵...")
sim_to_gallery = cosine_similarity(query_feats, gallery_feats)

# ------------------------ 7️⃣ 计算评分 ------------------------
print("\n计算open-set评分...")
scores_max = sim_to_gallery.max(axis=1)
scores_top5 = score_topk(sim_to_gallery, k=5)

# ------------------------ 8️⃣ 计算指标 ------------------------
print("\nRGB评分结果:")
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
    "gallery_size": int(gallery_feats.shape[0]),
    "query_size": int(query_feats.shape[0]),
    "known_queries": int(query_is_known.sum()),
    "unknown_queries": int((query_is_known==0).sum()),
    "metrics": metrics_results
}

metrics_path = os.path.join(save_dir, "rgb_results.json")
with open(metrics_path, "w") as f:
    json.dump(output, f, indent=4)

print("\nResults saved:", metrics_path)
print("\n✅ RGB scoring实验完成！")