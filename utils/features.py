import numpy as np


def _l2_normalize(feat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def load_feature(feat_path, multi_view=None, aggregation=None):
    """
    Load a feature file.

    aggregation:
    - "mean": mean-pool multi-view features into one object vector
    - "none": keep the saved tensor shape

    The legacy multi_view flag remains supported for backward compatibility.
    """
    feat = np.load(feat_path).astype(np.float32)

    if aggregation is None:
        if multi_view is None:
            aggregation = "mean"
        else:
            aggregation = "mean" if multi_view else "none"

    if aggregation == "mean":
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
