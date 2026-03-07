# scripts/run_depth.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from utils.metrics import compute_open_set_metrics
from utils.split import build_class_file_dict, build_open_set_split
from utils.features import load_feature
from configs.exp_config import DEPTH_FEAT_DIR, RESULT_DIR, KNOWN_NUM, UNKNOWN_NUM, GALLERY_RATIO, SEED
from utils.scoring import score_topk, score_margin, score_energy

# ------------------------
# 0️⃣ 固定随机种子
# ------------------------
random.seed(SEED)
np.random.seed(SEED)

# ------------------------
# 1️⃣ 结果目录
# ------------------------
save_dir = os.path.join(RESULT_DIR, "depth_scoring")
os.makedirs(save_dir, exist_ok=True)

# ------------------------
# 2️⃣ 构建类别 → 文件列表
# ------------------------
class_to_files = build_class_file_dict(DEPTH_FEAT_DIR)

# ------------------------
# 3️⃣ Open-set split
# ------------------------
(
    gallery_file_list,
    query_file_list,
    query_is_known,
    KNOWN_CLASSES,
    UNKNOWN_CLASSES,
) = build_open_set_split(class_to_files, known_num=KNOWN_NUM, unknown_num=UNKNOWN_NUM, gallery_ratio=GALLERY_RATIO, seed=SEED)

print("Known classes:", KNOWN_CLASSES)
print("Unknown classes:", UNKNOWN_CLASSES)

# ------------------------
# 4️⃣ 加载特征
# ------------------------
gallery_feats = [load_feature(f, multi_view=False) for cls, f in gallery_file_list]
query_feats = [load_feature(f, multi_view=False) for cls, f in query_file_list]
query_is_known = np.array(query_is_known)

gallery_feats = np.stack(gallery_feats)
query_feats = np.stack(query_feats)

# ------------------------
# 5️⃣ L2归一化
# ------------------------
gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)
query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)

print("Gallery size:", gallery_feats.shape)
print("Query size:", query_feats.shape)
print("Known queries:", query_is_known.sum(), "Unknown queries:", (query_is_known==0).sum())



# ------------------------
# 7️⃣ 计算相似度矩阵
# ------------------------
print("\n📊 计算相似度矩阵...")
sim_to_gallery = cosine_similarity(query_feats, gallery_feats)

# ------------------------
# 8️⃣ 计算四种评分
# ------------------------
scores_max = sim_to_gallery.max(axis=1)
scores_top5 = score_topk(sim_to_gallery, k=5)
scores_margin = score_margin(sim_to_gallery)
scores_energy = score_energy(sim_to_gallery)

# ------------------------
# 9️⃣ 计算指标
# ------------------------
methods = [
    ("Max", scores_max),
    ("Top-5 Mean", scores_top5),
    ("Margin", scores_margin),
    ("Energy", scores_energy)
]

results = []
for name, scores in methods:
    auroc, fpr95 = compute_open_set_metrics(scores, query_is_known, tpr_target=0.95)
    results.append({'method': name, 'auroc': auroc, 'fpr95': fpr95})
    print(f"{name:12} AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")

# ------------------------
# 🔟 可视化
# ------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, (name, scores) in enumerate(methods):
    ax = axes[idx]
    ax.hist(scores[query_is_known == 1], bins=50, alpha=0.6, label='Known', density=True)
    ax.hist(scores[query_is_known == 0], bins=50, alpha=0.6, label='Unknown', density=True)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{name}\nAUROC={results[idx]["auroc"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "scoring_comparison_depth.png"), dpi=200, bbox_inches="tight")
plt.show()

# ------------------------
# 1️⃣1️⃣ 保存结果
# ------------------------
np.savez(
    os.path.join(save_dir, f"depth_scoring_comparison_seed{SEED}.npz"),
    scores_max=scores_max,
    scores_top5=scores_top5,
    scores_margin=scores_margin,
    scores_energy=scores_energy,
    query_is_known=query_is_known,
    results=results,
    known_classes=KNOWN_CLASSES,
    unknown_classes=UNKNOWN_CLASSES
)

print("\n✅ Depth scoring实验完成！")