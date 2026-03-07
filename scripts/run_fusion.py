# scripts/run_fusion_final.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from utils.metrics import compute_open_set_metrics
from utils.split import build_class_file_dict, build_open_set_split
from utils.features import load_feature
from utils.scoring import score_max, score_topk

# ======================
# 0️⃣ 固定随机种子
# ======================
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# ======================
# 1️⃣ 路径配置
# ======================
rgb_root = r"D:/1Ahaha/AA3d/output_224_clip_feat"
depth_root = r"D:/1Ahaha/AA3d/output_224_clip_feat_depth"
save_dir = os.path.join("results", "fusion_final")
os.makedirs(save_dir, exist_ok=True)

# ======================
# 2️⃣ 构建公共文件列表
# ======================
class_to_files = {}
rgb_dict = build_class_file_dict(rgb_root)
depth_dict = build_class_file_dict(depth_root)

for cls in rgb_dict:
    if cls not in depth_dict:
        continue
    rgb_files = set([os.path.basename(f) for f in rgb_dict[cls]])
    depth_files = set([os.path.basename(f) for f in depth_dict[cls]])
    common_files = sorted(list(rgb_files & depth_files))
    if len(common_files) > 0:
        class_to_files[cls] = common_files

# ======================
# 3️⃣ Open-set split
# ======================
(
    gallery_file_list,
    query_file_list,
    query_is_known,
    KNOWN_CLASSES,
    UNKNOWN_CLASSES,
) = build_open_set_split(class_to_files, known_num=40, unknown_num=10, gallery_ratio=0.7, seed=SEED)

query_is_known = np.array(query_is_known)

print("Known classes:", KNOWN_CLASSES)
print("Unknown classes:", UNKNOWN_CLASSES)
print("Gallery size:", len(gallery_file_list))
print("Query size:", len(query_file_list))
print("Known queries:", query_is_known.sum(), "Unknown queries:", (query_is_known==0).sum())

# ======================
# 4️⃣ 融合函数
# ======================
def fuse_feature(rgb_feat, depth_feat, alpha=0.3):
    fused = alpha * rgb_feat + (1 - alpha) * depth_feat
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused

# ======================
# 5️⃣ 加载特征
# ======================
def load_feats(file_list, alpha=0.3):
    feats = []
    for cls, fname in file_list:
        rgb_feat = load_feature(os.path.join(rgb_root, cls, fname), multi_view=False)
        depth_feat = load_feature(os.path.join(depth_root, cls, fname), multi_view=True)
        feats.append(fuse_feature(rgb_feat, depth_feat, alpha=alpha))
    return np.stack(feats)

gallery_feats = load_feats(gallery_file_list, alpha=0.3)
query_feats = load_feats(query_file_list, alpha=0.3)

# ======================
# 6️⃣ Open-set Scoring
# ======================
sim = cosine_similarity(query_feats, gallery_feats)
scores_max = score_max(sim)
scores_top5 = score_topk(sim, k=5)

# ======================
# 7️⃣ 指标
# ======================
auroc_max, fpr95_max = compute_open_set_metrics(scores_max, query_is_known)
auroc_top5, fpr95_top5 = compute_open_set_metrics(scores_top5, query_is_known)

print(f"\nFusion α=0.3 Max AUROC: {auroc_max:.4f}, FPR95: {fpr95_max:.4f}")
print(f"Fusion α=0.3 Top-5 AUROC: {auroc_top5:.4f}, FPR95: {fpr95_top5:.4f}")

# ======================
# 8️⃣ 可视化分数分布
# ======================
plt.figure(figsize=(10, 6))

plt.hist(scores_max[query_is_known==1], bins=50, alpha=0.6, density=True, label='Known Query (Max)')
plt.hist(scores_max[query_is_known==0], bins=50, alpha=0.6, density=True, label='Unknown Query (Max)')

plt.hist(scores_top5[query_is_known==1], bins=50, alpha=0.6, density=True, label='Known Query (Top-5 Mean)')
plt.hist(scores_top5[query_is_known==0], bins=50, alpha=0.6, density=True, label='Unknown Query (Top-5 Mean)')

plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Fusion α=0.3 Open-set Score Distribution")
plt.legend()
plt.grid(True, alpha=0.3)

plot_path = os.path.join(save_dir, "fusion_alpha0.3_score_distribution.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved plot to: {plot_path}")

# ======================
# 9️⃣ 保存结果
# ======================
np.savez(
    os.path.join(save_dir, "fusion_alpha0.3_results.npz"),
    scores_max=scores_max,
    scores_top5=scores_top5,
    query_is_known=query_is_known,
    auroc_max=auroc_max,
    fpr95_max=fpr95_max,
    auroc_top5=auroc_top5,
    fpr95_top5=fpr95_top5,
    known_classes=KNOWN_CLASSES,
    unknown_classes=UNKNOWN_CLASSES
)

print("\n✅ Fusion α=0.3 scoring experiment done!")