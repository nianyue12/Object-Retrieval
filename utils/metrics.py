import numpy as np
from tqdm import tqdm


def _as_label_arrays(gallery_labels, query_labels):
    return np.array(gallery_labels), np.array(query_labels)


def similarity_to_distance(sim_matrix):
    # Features are L2-normalized before retrieval, so cosine distance is 1 - cosine similarity.
    return 1.0 - np.asarray(sim_matrix, dtype=np.float32)


def compute_map(sim_matrix, gallery_labels, query_labels):
    gallery_labels, query_labels = _as_label_arrays(gallery_labels, query_labels)

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
    gallery_labels, query_labels = _as_label_arrays(gallery_labels, query_labels)

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
    gallery_labels, query_labels = _as_label_arrays(gallery_labels, query_labels)

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


def compute_legacy_metrics(sim_matrix, gallery_labels, query_labels):
    return {
        "mAP": compute_map(sim_matrix, gallery_labels, query_labels),
        "NDCG": compute_ndcg(sim_matrix, gallery_labels, query_labels),
        "ANMRR": compute_anmrr(sim_matrix, gallery_labels, query_labels),
    }


def compute_hgm2r_map(dist_matrix, query_labels, gallery_labels, top_k=None):
    n_gallery = dist_matrix.shape[1]
    if top_k is None:
        top_k = n_gallery
    top_k = min(top_k, n_gallery)
    sorted_indices = dist_matrix.argsort(axis=1)
    results = []

    for q in tqdm(range(len(query_labels)), desc="Computing HGM2R mAP"):
        order = sorted_indices[q]
        precision_list = []
        hits = 0
        for rank in range(top_k):
            if query_labels[q] == gallery_labels[order[rank]]:
                hits += 1
                precision_list.append(hits / (rank + 1))
        if hits > 0:
            for idx in range(len(precision_list)):
                precision_list[idx] = max(precision_list[idx:])
            results.append(float(np.mean(precision_list)))
        else:
            results.append(0.0)

    return float(np.mean(results))


def compute_hgm2r_recall(dist_matrix, query_labels, gallery_labels, top_k=100):
    n_gallery = dist_matrix.shape[1]
    top_k = min(top_k, n_gallery)
    sorted_indices = dist_matrix.argsort(axis=1)
    results = []

    for q in tqdm(range(len(query_labels)), desc="Computing HGM2R Recall@100"):
        order = sorted_indices[q]
        hits = 0
        for rank in range(top_k):
            if query_labels[q] == gallery_labels[order[rank]]:
                hits += 1
        # This follows HGM2R's released evaluator exactly, even though the denominator is unusual.
        results.append(hits / max(1, np.sum(query_labels == query_labels[q])))

    return float(np.mean(results))


def compute_hgm2r_ndcg(dist_matrix, query_labels, gallery_labels, k=100):
    n_gallery = dist_matrix.shape[1]
    k = min(k, n_gallery)
    sorted_indices = dist_matrix.argsort(axis=1)
    results = []

    for q in tqdm(range(len(query_labels)), desc="Computing HGM2R NDCG@100"):
        order = sorted_indices[q]
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, n_gallery + 2)))
        dcg = np.cumsum(
            [
                1.0 / np.log2(rank + 2) if query_labels[q] == gallery_labels[item] else 0.0
                for rank, item in enumerate(order)
            ]
        )
        results.append(float((dcg / idcg)[k - 1]))

    return float(np.mean(results))


def compute_hgm2r_anmrr(dist_matrix, query_labels, gallery_labels):
    query_labels = np.array(query_labels)
    gallery_labels = np.array(gallery_labels)
    n_query = dist_matrix.shape[0]
    ng = np.array([(query_labels[q] == gallery_labels).sum() for q in range(n_query)])
    sorted_indices = dist_matrix.argsort(axis=1)
    results = []

    for q in tqdm(range(n_query), desc="Computing HGM2R ANMRR"):
        cur_ng = int(ng[q])
        if cur_ng <= 0:
            results.append(0.0)
            continue
        cutoff = min(4 * cur_ng, 2 * int(ng.max()))
        order = sorted_indices[q]
        arr = np.sum(
            [
                (rank + 1) / cur_ng
                if query_labels[q] == gallery_labels[order[rank]]
                else (cutoff + 1) / cur_ng
                for rank in range(cur_ng)
            ]
        )
        mrr = arr - 0.5 * cur_ng - 0.5
        nmrr = mrr / (cutoff - 0.5 * cur_ng + 0.5)
        results.append(float(nmrr))

    return float(np.mean(results))


def compute_hgm2r_metrics(sim_matrix, gallery_labels, query_labels):
    gallery_labels, query_labels = _as_label_arrays(gallery_labels, query_labels)
    dist_matrix = similarity_to_distance(sim_matrix)
    return {
        "mAP": compute_hgm2r_map(dist_matrix, query_labels, gallery_labels),
        "NDCG@100": compute_hgm2r_ndcg(dist_matrix, query_labels, gallery_labels, k=100),
        "ANMRR": compute_hgm2r_anmrr(dist_matrix, query_labels, gallery_labels),
        "Recall@100": compute_hgm2r_recall(
            dist_matrix, query_labels, gallery_labels, top_k=100
        ),
    }


def evaluate_retrieval(sim_matrix, gallery_labels, query_labels, metric_style="hgm2r"):
    metric_style = metric_style.lower()
    if metric_style not in {"legacy", "hgm2r", "both"}:
        raise ValueError(f"Unsupported metric_style: {metric_style}")

    metrics_by_style = {}
    if metric_style in {"legacy", "both"}:
        metrics_by_style["legacy"] = compute_legacy_metrics(
            sim_matrix, gallery_labels, query_labels
        )
    if metric_style in {"hgm2r", "both"}:
        metrics_by_style["hgm2r"] = compute_hgm2r_metrics(
            sim_matrix, gallery_labels, query_labels
        )

    primary_style = "hgm2r" if metric_style in {"hgm2r", "both"} else "legacy"
    return {
        "primary_style": primary_style,
        "metrics": metrics_by_style[primary_style],
        "metrics_by_style": metrics_by_style,
    }


def format_metric_report(metrics):
    parts = []
    for key, value in metrics.items():
        if np.isfinite(value):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}=nan")
    return ", ".join(parts)
