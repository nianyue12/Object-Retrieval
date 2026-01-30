import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve

# ======================
# 1️⃣ 路径配置
# ======================
feat_root = r"D:/1Ahaha/AA3d/output_224_clip_feat"
data_root = r"D:/1Ahaha/AA3d/output_224"
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)


# ======================
# 2️⃣ 类别划分（Seen / Unseen）
# ======================
def split_classes(data_root, known_ratio=0.7, seed=0):
    all_classes = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]
    all_classes.sort()

    np.random.seed(seed)
    np.random.shuffle(all_classes)

    k = int(len(all_classes) * known_ratio)
    return all_classes[:k], all_classes[k:]


KNOWN_CLASSES, UNKNOWN_CLASSES = split_classes(
    data_root, known_ratio=0.75, seed=0
)

print("Known classes:", KNOWN_CLASSES)
print("Unknown classes:", UNKNOWN_CLASSES)

# ======================
# 3️⃣ & 4️⃣ 构建 Gallery 和 Query（修复数据泄露）
# ======================
random.seed(0)
GALLERY_RATIO = 0.7  # 70% models → gallery

gallery_feats = []
gallery_labels = []

query_feats = []
query_is_known = []   # 1 = known, 0 = unknown

# ======================
# 3A. 已知类：模型级 split
# ======================
for cls in KNOWN_CLASSES:
    cls_path = os.path.join(data_root, cls)
    cls_name = cls.replace("_multi_view", "")

    obj_ids = os.listdir(cls_path)
    random.shuffle(obj_ids)

    split_idx = int(len(obj_ids) * GALLERY_RATIO)
    gallery_ids = obj_ids[:split_idx]
    query_ids   = obj_ids[split_idx:]

    # Gallery
    for obj_id in gallery_ids:
        feat_path = os.path.join(feat_root, f"{cls_name}_{obj_id}.npy")
        if not os.path.exists(feat_path):
            continue
        gallery_feats.append(np.load(feat_path).squeeze())
        gallery_labels.append(cls_name)

    # Known Query（⚠ 不再和 gallery 重叠）
    for obj_id in query_ids:
        feat_path = os.path.join(feat_root, f"{cls_name}_{obj_id}.npy")
        if not os.path.exists(feat_path):
            continue
        query_feats.append(np.load(feat_path).squeeze())
        query_is_known.append(1)   # known

# ======================
# 3B. 未知类：全部作为 Query
# ======================
for cls in UNKNOWN_CLASSES:
    cls_path = os.path.join(data_root, cls)
    cls_name = cls.replace("_multi_view", "")

    for obj_id in os.listdir(cls_path):
        feat_path = os.path.join(feat_root, f"{cls_name}_{obj_id}.npy")
        if not os.path.exists(feat_path):
            continue
        
        query_feats.append(np.load(feat_path).squeeze())
        query_is_known.append(0)   # unknown ← 这是关键！

# ======================
# 4️⃣ 转换为numpy数组
# ======================
gallery_feats = np.stack(gallery_feats)
gallery_labels = np.array(gallery_labels)

query_feats = np.stack(query_feats)
query_is_known = np.array(query_is_known)

print("Gallery size:", gallery_feats.shape)
print("Query size:", query_feats.shape)
print("Known queries:", query_is_known.sum(),
      "Unknown queries:", (query_is_known == 0).sum())

# ======================
# 5️⃣ Open-set Score 计算
# ======================
# Open-set score = max similarity to gallery
sim_qg = cosine_similarity(query_feats, gallery_feats)
open_scores = sim_qg.max(axis=1)


# ======================
# 6️⃣ 开集指标（AUROC / FPR95）
# ======================
auroc = roc_auc_score(query_is_known, open_scores)

fpr, tpr, thresholds = roc_curve(query_is_known, open_scores)
idx_95 = np.argmin(np.abs(tpr - 0.95))
fpr95 = fpr[idx_95]

print(f"AUROC: {auroc:.4f}")
print(f"FPR@95TPR: {fpr95:.4f}")

# ======================
# 7️⃣ 可视化（合理版本）
# ======================
kk_sim = cosine_similarity(gallery_feats, gallery_feats)
kk_sim = kk_sim[~np.eye(len(kk_sim), dtype=bool)]

plt.figure(figsize=(8, 5))

plt.hist(
    kk_sim.flatten(),
    bins=100,
    density=True,
    alpha=0.6,
    label="Known → Known (all pairs)"
)

plt.hist(
    open_scores[query_is_known == 1],
    bins=100,
    density=True,
    alpha=0.6,
    label="Known Query → Known (max)"
)

plt.hist(
    open_scores[query_is_known == 0],
    bins=100,
    density=True,
    alpha=0.6,
    label="Unknown Query → Known (max)"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Open-set Similarity Distribution (CLIP-MV RGB)")
plt.legend()
plt.grid(True)

plot_path = os.path.join(save_dir, "open_set_similarity_distribution.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {plot_path}")

# ======================
# 8️⃣ 保存结果（为论文 / 多次实验兜底）
# ======================
np.savez(
    os.path.join(save_dir, "open_set_results_seed0.npz"),
    open_scores=open_scores,
    query_is_known=query_is_known,
    auroc=auroc,
    fpr95=fpr95,
    known_classes=KNOWN_CLASSES,
    unknown_classes=UNKNOWN_CLASSES
)