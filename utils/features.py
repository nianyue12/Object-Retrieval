import numpy as np


def _l2_normalize(feat: np.ndarray) -> np.ndarray:
    """对单个特征向量做 L2 归一化。"""
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def load_feature(feat_path, multi_view=None, aggregation=None):
    """
    功能：读取 `.npy` 特征文件，并按需要做聚合。

    参数：
        feat_path: 特征文件路径
        aggregation:
            - "mean": 对多视图特征做均值池化
            - "none": 保持原始张量形状
        multi_view:
            兼容旧接口，内部会转成 aggregation 逻辑

    返回：
        聚合后或原始的特征数组
    """
    feat = np.load(feat_path).astype(np.float32)

    # 兼容旧版 multi_view 参数
    if aggregation is None:
        if multi_view is None:
            aggregation = "mean"
        else:
            aggregation = "mean" if multi_view else "none"

    if aggregation == "mean":
        # 多视图特征默认做均值池化，再归一化成单个物体向量
        if feat.ndim == 2:
            feat = feat.mean(axis=0)
            feat = _l2_normalize(feat)
        elif feat.ndim == 1:
            feat = _l2_normalize(feat)
        else:
            raise ValueError(
                f"Unsupported feature shape for mean aggregation: {feat.shape}"
            )
    elif aggregation == "none":
        pass
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggregation}")

    return feat.squeeze()
