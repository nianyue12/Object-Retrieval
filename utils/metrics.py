# utils/metrics.py
import numpy as np
from tqdm import tqdm


# ------------------------------------------------
# mAP
# ------------------------------------------------
def compute_map(sim_matrix, gallery_labels, query_labels):

    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)

    Q = len(query_labels)

    AP_list = []

    for q in tqdm(range(Q), desc="Computing mAP"):

        sims = sim_matrix[q]

        ranking = np.argsort(-sims)
        sorted_labels = gallery_labels[ranking]

        matches = (sorted_labels == query_labels[q]).astype(np.int32)

        if matches.sum() == 0:
            AP_list.append(0)
            continue

        cum_hits = np.cumsum(matches)

        precision = cum_hits / (np.arange(len(matches)) + 1)

        AP = np.sum(precision * matches) / matches.sum()

        AP_list.append(AP)

    return np.mean(AP_list)


# ------------------------------------------------
# NDCG
# ------------------------------------------------
def compute_ndcg(sim_matrix, gallery_labels, query_labels):

    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)

    Q = len(query_labels)

    ndcg_list = []

    for q in tqdm(range(Q), desc="Computing NDCG"):

        sims = sim_matrix[q]

        ranking = np.argsort(-sims)
        sorted_labels = gallery_labels[ranking]

        rel = (sorted_labels == query_labels[q]).astype(np.int32)

        if rel.sum() == 0:
            ndcg_list.append(0)
            continue

        discounts = 1 / np.log2(np.arange(len(rel)) + 2)

        dcg = np.sum(rel * discounts)

        ideal_rel = np.sort(rel)[::-1]

        idcg = np.sum(ideal_rel * discounts)

        ndcg = dcg / idcg

        ndcg_list.append(ndcg)

    return np.mean(ndcg_list)


# ------------------------------------------------
# ANMRR  (MPEG-7 标准)
# ------------------------------------------------
def compute_anmrr(sim_matrix, gallery_labels, query_labels, K=100):

    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)

    Q = len(query_labels)
    NMRR_list = []

    for q in tqdm(range(Q), desc="Computing ANMRR"):

        sims = sim_matrix[q]

        ranking = np.argsort(-sims)
        sorted_labels = gallery_labels[ranking]

        NG = np.sum(gallery_labels == query_labels[q])

        if NG == 0:
            continue

        Kq = min(4 * NG, K)

        # relevant positions
        relevant = np.where(sorted_labels == query_labels[q])[0] + 1

        # penalty
        ranks = np.minimum(relevant, 1.25 * Kq)

        AVR = np.mean(ranks)

        NMRR = (AVR - 0.5 * (1 + NG)) / (1.25 * Kq - 0.5 * (1 + NG))

        NMRR_list.append(NMRR)

    return np.mean(NMRR_list)