import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import (
    ALPHA_FUSION,
    BATCH_SIZE,
    DEFAULT_PROTOCOL_PATH,
    DEPTH_FEAT_DIR,
    RGB_FEAT_DIR,
    UNSEEN_RESULT_DIR,
)
from utils.features import load_feature
from utils.metrics import compute_anmrr, compute_map, compute_ndcg
from utils.protocol import get_split_items, load_protocol, materialize_split_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot retrieval on unseen classes."
    )
    parser.add_argument("--mode", choices=["rgb", "depth", "fusion"], required=True)
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--alpha", type=float, default=ALPHA_FUSION)
    return parser.parse_args()


def load_single_modality(protocol, split_name, feat_root):
    entries = materialize_split_paths(protocol, split_name, feat_root)
    feats, labels = [], []
    for cls, path in entries:
        feats.append(load_feature(path, multi_view=False))
        labels.append(cls)
    return np.stack(feats), np.array(labels)


def load_fusion(protocol, split_name, alpha):
    feats, labels = [], []
    for cls, item in get_split_items(protocol, split_name):
        rgb_feat = load_feature(os.path.join(RGB_FEAT_DIR, cls, item), multi_view=False)
        depth_feat = load_feature(os.path.join(DEPTH_FEAT_DIR, cls, item), multi_view=False)
        fused = alpha * rgb_feat + (1.0 - alpha) * depth_feat
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        feats.append(fused)
        labels.append(cls)
    return np.stack(feats), np.array(labels)


def normalize_rows(feats):
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def compute_similarity(query_feats, gallery_feats):
    sim = np.zeros((query_feats.shape[0], gallery_feats.shape[0]), dtype=np.float32)
    for start in tqdm(range(0, query_feats.shape[0], BATCH_SIZE), desc="Computing similarity"):
        end = min(start + BATCH_SIZE, query_feats.shape[0])
        sim[start:end] = query_feats[start:end] @ gallery_feats.T
    return sim


def main():
    args = parse_args()
    protocol = load_protocol(args.protocol)

    if args.mode == "rgb":
        gallery_feats, gallery_labels = load_single_modality(
            protocol, "gallery_unseen", RGB_FEAT_DIR
        )
        query_feats, query_labels = load_single_modality(
            protocol, "query_unseen", RGB_FEAT_DIR
        )
    elif args.mode == "depth":
        gallery_feats, gallery_labels = load_single_modality(
            protocol, "gallery_unseen", DEPTH_FEAT_DIR
        )
        query_feats, query_labels = load_single_modality(
            protocol, "query_unseen", DEPTH_FEAT_DIR
        )
    else:
        gallery_feats, gallery_labels = load_fusion(protocol, "gallery_unseen", args.alpha)
        query_feats, query_labels = load_fusion(protocol, "query_unseen", args.alpha)

    gallery_feats = normalize_rows(gallery_feats)
    query_feats = normalize_rows(query_feats)

    print(f"Seen classes used for training protocol: {protocol['seen_classes']}")
    print(f"Unseen classes used for retrieval: {protocol['unseen_classes']}")
    print(f"Gallery size: {gallery_feats.shape[0]}")
    print(f"Query size: {query_feats.shape[0]}")

    sim_matrix = compute_similarity(query_feats, gallery_feats)

    m_ap = compute_map(sim_matrix, gallery_labels, query_labels)
    ndcg = compute_ndcg(sim_matrix, gallery_labels, query_labels)
    anmrr = compute_anmrr(sim_matrix, gallery_labels, query_labels)

    os.makedirs(UNSEEN_RESULT_DIR, exist_ok=True)
    if args.mode == "fusion":
        alpha_tag = f"{args.alpha:.2f}".replace(".", "p")
        save_name = f"{args.mode}_zero_shot_alpha{alpha_tag}.json"
    else:
        save_name = f"{args.mode}_zero_shot.json"
    save_path = os.path.join(UNSEEN_RESULT_DIR, save_name)

    output = {
        "mode": args.mode,
        "protocol_path": args.protocol,
        "seen_classes": protocol["seen_classes"],
        "unseen_classes": protocol["unseen_classes"],
        "gallery_size": int(gallery_feats.shape[0]),
        "query_size": int(query_feats.shape[0]),
        "alpha_fusion": float(args.alpha),
        "metrics": {
            "mAP": float(m_ap),
            "NDCG": float(ndcg),
            "ANMRR": float(anmrr),
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Saved: {save_path}")
    print(
        f"{args.mode} zero-shot unseen retrieval: "
        f"mAP={m_ap:.4f}, NDCG={ndcg:.4f}, ANMRR={anmrr:.4f}"
    )


if __name__ == "__main__":
    main()
