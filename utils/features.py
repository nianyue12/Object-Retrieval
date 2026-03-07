# utils/features.py
import numpy as np

def load_feature(feat_path, multi_view=True):
    """
    加载特征文件
    - multi_view: 如果是多视图特征，沿 axis=0 平均
    """
    feat = np.load(feat_path)
    
    if multi_view and len(feat.shape) == 2:
        feat = feat.mean(axis=0)
    
    feat = feat.squeeze()
    return feat