import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# 1️⃣ 路径配置
# ======================
feat_root = r"D:/1Ahaha/AA3d/output_224_clip_feat"
label_root = r"D:/1Ahaha/AA3d/output_224"  # 用目录名当类别

# ======================
# 2️⃣ 读取所有特征
# ======================
features = []
labels = []
ids = []

for category in os.listdir(label_root):
    cat_path = os.path.join(label_root, category)
    if not os.path.isdir(cat_path):
        continue

    for obj_id in os.listdir(cat_path):
        feat_path = os.path.join(feat_root, f"{obj_id}.npy")
        if not os.path.exists(feat_path):
            continue

        feat = np.load(feat_path)  # (1, 512)
        features.append(feat.squeeze())
        labels.append(category)
        ids.append(obj_id)

features = np.stack(features)  # (N, 512)
labels = np.array(labels)

print("Loaded features:", features.shape)


# ======================
# 3️⃣ 计算 cosine similarity
# ======================
sim_matrix = cosine_similarity(features, features)  # (N, N)

# ======================
# 4️⃣ Top-1 检索验证
# ======================
correct = 0
total = len(features)

for i in range(total):
    # 排除自己
    sim_scores = sim_matrix[i]
    sim_scores[i] = -1

    top1_idx = np.argmax(sim_scores)
    if labels[top1_idx] == labels[i]:
        correct += 1

print(f"Top-1 Accuracy: {correct / total:.4f}")
