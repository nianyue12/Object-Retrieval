import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 加载 CLIP 模型（ViT-B/32，GPU/CPU自动适配）
model, preprocess = clip.load("ViT-B/32", device=device)

# 准备测试图片和文本
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# 无梯度提取特征（避免显存占用）
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# 打印特征形状（核心验收结果）
print(image_features.shape, text_features.shape)