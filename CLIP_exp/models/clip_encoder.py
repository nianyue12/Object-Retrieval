# models/clip_encoder.py

import torch
import clip
from PIL import Image
import numpy as np


class CLIPEncoder:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, img: Image.Image):
        """
        img: PIL Image (RGB)
        return: (512,) numpy feature
        """
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    def encode_multi_view(self, imgs):
        """
        imgs: list[PIL.Image]
        return: (512,) numpy feature (mean pooled)
        """
        feats = [self.encode_image(img) for img in imgs]
        feats = np.stack(feats)
        return feats.mean(axis=0)
