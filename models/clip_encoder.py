import numpy as np
import torch
from PIL import Image

from utils.clip_utils import load_clip_model


class CLIPEncoder:
    """
    功能：对 CLIP 图像编码流程做一层简单封装。

    用途：
        统一处理单张图像编码和多视图平均池化，
        方便特征提取脚本直接调用。
    """

    def __init__(self, model_name="ViT-B/32", device=None):
        # device 为空时自动优先使用 CUDA，保持各个特征提取脚本调用简单。
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _, self.model, self.preprocess = load_clip_model(
            model_name,
            device=self.device,
        )

    @torch.no_grad()
    def encode_image(self, img: Image.Image):
        """
        功能：编码单张 RGB 图像。

        参数：
            img: PIL 格式的 RGB 图像

        返回：
            单张图像的归一化特征向量
        """
        # 先做 CLIP 预处理，再补 batch 维度送入模型
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_tensor)
        # CLIP 特征归一化后，后续点积就等价于余弦相似度。
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    def encode_multi_view(self, imgs, aggregate=True):
        """
        功能：编码多视图图像，并按需做平均池化。

        参数：
            imgs: 多张 PIL.Image 组成的列表
            aggregate: 是否把多视图特征融合成一个向量

        返回：
            aggregate=True  -> 单个物体的融合特征
            aggregate=False -> 每个视图各自的特征
        """
        feats = [self.encode_image(img) for img in imgs]
        feats = np.stack(feats).astype(np.float32)

        if not aggregate:
            return feats

        # 默认对多视图特征做均值池化，再做一次归一化
        fused_feat = feats.mean(axis=0)
        # 多视图平均会改变向量长度，因此融合后需要重新归一化。
        fused_feat = fused_feat / np.linalg.norm(fused_feat)
        return fused_feat
