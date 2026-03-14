import numpy as np
from tqdm import tqdm


def compute_map(sim_matrix, gallery_labels, query_labels):
    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)

    ap_list = []
    for q in tqdm(range(len(query_labels)), desc="Computing mAP"):
        sims = sim_matrix[q]
        ranking = np.argsort(-sims)
        sorted_labels = gallery_labels[ranking]

        matches = (sorted_labels == query_labels[q]).astype(np.int32)
        if matches.sum() == 0:
            ap_list.append(0.0)
            continue

        cum_hits = np.cumsum(matches)
        precision = cum_hits / (np.arange(len(matches)) + 1)
        ap = np.sum(precision * matches) / matches.sum()
        ap_list.append(float(ap))

    return float(np.mean(ap_list))


def compute_ndcg(sim_matrix, gallery_labels, query_labels):
    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)

    ndcg_list = []
    for q in tqdm(range(len(query_labels)), desc="Computing NDCG"):
        sims = sim_matrix[q]
        ranking = np.argsort(-sims)
        sorted_labels = gallery_labels[ranking]

        rel = (sorted_labels == query_labels[q]).astype(np.int32)
        if rel.sum() == 0:
            ndcg_list.append(0.0)
            continue

        discounts = 1.0 / np.log2(np.arange(len(rel)) + 2)
        dcg = np.sum(rel * discounts)
        ideal_rel = np.sort(rel)[::-1]
        idcg = np.sum(ideal_rel * discounts)
        ndcg = dcg / idcg
        ndcg_list.append(float(ndcg))

    return float(np.mean(ndcg_list))


def compute_anmrr(sim_matrix, gallery_labels, query_labels):
    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)

    unique_query_labels = np.unique(query_labels)
    ng_dict = {
        cls: int(np.sum(gallery_labels == cls))
        for cls in unique_query_labels
    }

    if len(ng_dict) == 0:
        return float("nan")

    gtm = max(ng_dict.values())
    nmrr_list = []

    for q in tqdm(range(len(query_labels)), desc="Computing ANMRR"):
        sims = sim_matrix[q]
        ranking = np.argsort(-sims)
        sorted_labels = gallery_labels[ranking]

        cls = query_labels[q]
        ng = ng_dict.get(cls, 0)
        if ng == 0:
            continue

        # MPEG-7 style adaptive cut-off for each query.
        kq = min(4 * ng, 2 * gtm)
        penalty_rank = 1.25 * kq
        relevant = np.where(sorted_labels == cls)[0] + 1
        ranks = np.minimum(relevant, penalty_rank)

        avr = np.mean(ranks)
        denom = penalty_rank - 0.5 * (1 + ng)
        if denom <= 0:
            continue

        nmrr = (avr - 0.5 * (1 + ng)) / denom
        nmrr = float(np.clip(nmrr, 0.0, 1.0))
        nmrr_list.append(nmrr)

    if len(nmrr_list) == 0:
        return float("nan")

    return float(np.mean(nmrr_list))
