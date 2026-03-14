# CLIP-based Open-set 3D Object Retrieval

main experiment protocol:

- train on `seen` categories only
- retrieve on `unseen` categories only
- evaluate generalized 3D retrieval performance on unseen classes

## Main Protocol

- Total categories used: `50`
- Seen classes: `10`
- Unseen classes: `40`
- Seen split: `80% train_seen`, `20% val_seen`
- Unseen split: `70% gallery_unseen`, `30% query_unseen`

The fixed protocol file is generated at:

- `configs/splits/shapenet_seen10_unseen40_seed0.json`

## Main Experiments

### 1. RGB-CLIP zero-shot unseen retrieval

- Input: 12-view RGB images per 3D object
- Encoder: CLIP `ViT-B/32`
- Evaluation: `query_unseen -> gallery_unseen`
- Metrics: `mAP`, `NDCG`, `ANMRR`

### 2. Depth-CLIP zero-shot unseen retrieval

- Input: point-cloud-derived depth maps
- Encoder: CLIP `ViT-B/32`
- Evaluation: `query_unseen -> gallery_unseen`
- Metrics: `mAP`, `NDCG`, `ANMRR`

### 3. RGB + Depth Fusion zero-shot unseen retrieval

- Fusion: `fused = alpha * rgb_feat + (1 - alpha) * depth_feat`
- Default fusion weight: `alpha = 0.5`
- Evaluation: `query_unseen -> gallery_unseen`
- Metrics: `mAP`, `NDCG`, `ANMRR`

## Current Main Results

| Method | mAP | NDCG | ANMRR |
| --- | ---: | ---: | ---: |
| RGB-CLIP | 0.5485 | 0.8770 | 0.3061 |
| Depth-CLIP | 0.4723 | 0.8455 | 0.3776 |
| RGB+Depth Fusion | 0.5673 | 0.8832 | 0.2904 |

Result files:

- `results/unseen_retrieval/rgb_zero_shot.json`
- `results/unseen_retrieval/depth_zero_shot.json`
- `results/unseen_retrieval/fusion_zero_shot_alpha0p50.json`

## Run

Build the fixed protocol:

```bash
python scripts/build_seen_unseen_protocol.py
```

Run unseen retrieval:

```bash
python scripts/run_unseen_retrieval.py --mode rgb
python scripts/run_unseen_retrieval.py --mode depth
python scripts/run_unseen_retrieval.py --mode fusion
```
