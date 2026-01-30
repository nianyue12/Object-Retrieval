import torch
import clip
from PIL import Image
import os
import numpy as np

# GPU / CPU 自动选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # 设置为推理模式

# ======== 改成你想测试的图片路径 ========
img_path = r"D:/1Ahaha/AA3d/output_224/bag_multi_view/1b9ef45fefefa35ed13f430b2941481/rgb_0000.png"

# 读图并预处理
img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

# 提取特征
with torch.no_grad():
    feat = model.encode_image(img)
    feat = feat / feat.norm(dim=-1, keepdim=True)  # 单位化

# 打印结果
print("CLIP 特征 shape:", feat.shape)  # [1, 512]

# 可选：保存成 numpy 文件
save_path = img_path.replace(".png", "_clip_feat.npy")
np.save(save_path, feat.cpu().numpy())
print(f"特征已保存到 {save_path}")
