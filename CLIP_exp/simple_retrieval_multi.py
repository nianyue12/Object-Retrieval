import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FEAT_DIR = r"D:/1Ahaha/AA3d/output_224_clip_feat"

features = []
labels = []

for fname in os.listdir(FEAT_DIR):
    if not fname.endswith(".npy"):
        continue
    feat = np.load(os.path.join(FEAT_DIR, fname)).reshape(-1)
    label = fname.split("_")[0]  # 从文件名取类别
    features.append(feat)
    labels.append(label)

features = np.stack(features)
labels = np.array(labels)

print("Loaded features:", features.shape)

sim = cosine_similarity(features, features)

def topk_acc(sim, labels, k):
    correct = 0
    for i in range(len(labels)):
        idx = np.argsort(-sim[i])
        idx = idx[1:k+1]  # 排除自己
        if labels[i] in labels[idx]:
            correct += 1
    return correct / len(labels)

print(f"Top-1 Accuracy: {topk_acc(sim, labels, 1):.4f}")
print(f"Top-5 Accuracy: {topk_acc(sim, labels, 5):.4f}")
