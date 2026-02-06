# CLIP_exp/utils/metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_open_set_metrics(scores, labels, tpr_target=0.95):
    """
    Compute open-set evaluation metrics.

    Args:
        scores (np.ndarray): shape (N,), higher = more likely known
        labels (np.ndarray): shape (N,), 1 = known, 0 = unknown
        tpr_target (float): target TPR, default 0.95

    Returns:
        auroc (float)
        fpr_at_tpr (float)
    """
    auroc = roc_auc_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - tpr_target))
    fpr_at_tpr = fpr[idx]

    return auroc, fpr_at_tpr
