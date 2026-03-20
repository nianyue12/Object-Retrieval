# CLIP-based Open-set 3D Object Retrieval

This project builds a ShapeNet-based open-set 3D object retrieval pipeline and studies a shared CLIP visual backbone for RGB views and point-cloud-derived depth maps.

Main pipeline:

- render `12` RGB views for each 3D object
- sample a `2048`-point point cloud for each object
- project the point cloud into `12` depth maps with rendered camera parameters
- extract pooled CLIP features for RGB views and depth maps
- evaluate unseen retrieval under a fixed `seen/unseen` protocol
- run zero-training visual-language retrieval on top of cached RGB-depth features

## Main Protocol

- Total categories used: `50`
- Seen classes: `10`
- Unseen classes: `40`
- Seen split: `80% train_seen`, `20% val_seen`
- Unseen split: `70% gallery_unseen`, `30% query_unseen`

Fixed protocol file:

- `configs/splits/shapenet_seen10_unseen40_seed0.json`

## Main Experiments

### 1. RGB-CLIP zero-shot unseen retrieval

- Input: `12` RGB views per 3D object
- Encoder: CLIP `ViT-B/32`
- Evaluation: `query_unseen -> gallery_unseen`
- Metrics: `mAP`, `NDCG@100`, `ANMRR`, `Recall@100`

### 2. Depth-CLIP zero-shot unseen retrieval

- Input: `12` point-cloud-derived depth maps per 3D object
- Encoder: CLIP `ViT-B/32`
- Evaluation: `query_unseen -> gallery_unseen`
- Metrics: `mAP`, `NDCG@100`, `ANMRR`, `Recall@100`

### 3. RGB + Depth Fusion zero-shot unseen retrieval

- Fusion: `fused = alpha * rgb_feat + (1 - alpha) * depth_feat`
- Default fusion weight: `alpha = 0.5`
- Evaluation: `query_unseen -> gallery_unseen`
- Metrics: `mAP`, `NDCG@100`, `ANMRR`, `Recall@100`

### 4. Visual-Language Retrieval

- Backbone: cached RGB CLIP features + cached depth CLIP features
- Text branch: CLIP text prototypes built from unseen class names with prompt ensembling
- Retrieval score: global blend of visual cosine similarity and text-posterior similarity
- No extra training: zero-training visual-language reranking on top of cached features
- Recommended default: `fusion + unseen text bank + prob similarity + global blend + semantic_weight=0.25`
- Metrics: `mAP`, `NDCG@100`, `ANMRR`, `Recall@100`

## Main Metric Style

- Main tables in this project now use the `HGM2R-style` evaluator.
- Primary reported metrics are `mAP`, `NDCG@100`, `ANMRR`, and `Recall@100`.
- The previous `legacy` evaluator is still available in code for optional compatibility checks, but it is no longer the default reporting style.

## Current Main Results

| Method | mAP | NDCG@100 | ANMRR | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| RGB-CLIP | 0.5512 | 0.7582 | 0.4577 | 0.2294 |
| Depth-CLIP | 0.4762 | 0.6896 | 0.5284 | 0.1805 |
| RGB+Depth Fusion | 0.5701 | 0.7709 | 0.4418 | 0.2363 |
| VL Retrieval | 0.6073 | 0.7837 | 0.4086 | 0.2476 |

Result files:

- `results/unseen_retrieval/rgb_zero_shot_both.json`
- `results/unseen_retrieval/depth_zero_shot_both.json`
- `results/unseen_retrieval/fusion_zero_shot_alpha0p50_both.json`
- `results/unseen_retrieval/vl_fusion_unseen_confidence_prob_global_blend_sw0p25_a0p50_hgm2r.json`

Compared with the zero-shot RGB+Depth baseline, VL Retrieval improves:

- `mAP`: `+0.0371`
- `NDCG@100`: `+0.0128`
- `ANMRR`: `-0.0332`
- `Recall@100`: `+0.0113`

## Run

Prepare data:

```bash
python tools/batch_render.py
python tools/batch_sample_pc_custom.py
python tools/pc_to_depth.py
```

Extract CLIP features:

```bash
python scripts/extract_rgb_features.py
python scripts/extract_depth_features.py
```

Build the fixed protocol:

```bash
python scripts/build_seen_unseen_protocol.py
```

Run zero-shot unseen retrieval:

```bash
python scripts/run_unseen_retrieval.py --mode rgb --metric_style hgm2r
python scripts/run_unseen_retrieval.py --mode depth --metric_style hgm2r
python scripts/run_unseen_retrieval.py --mode fusion --metric_style hgm2r
```

Run zero-training visual-language retrieval:

```bash
python scripts/run_vl_retrieval.py --mode fusion --metric_style hgm2r
```

If you need both the main `HGM2R-style` metrics and the older compatibility metrics in the same file, use:

```bash
python scripts/run_unseen_retrieval.py --mode fusion --metric_style both
python scripts/run_vl_retrieval.py --mode fusion --metric_style both
```

Key outputs:

- `results/unseen_retrieval/rgb_zero_shot_hgm2r.json` or `rgb_zero_shot_both.json`
- `results/unseen_retrieval/depth_zero_shot_hgm2r.json` or `depth_zero_shot_both.json`
- `results/unseen_retrieval/fusion_zero_shot_alpha0p50_hgm2r.json` or `fusion_zero_shot_alpha0p50_both.json`
- `results/unseen_retrieval/vl_fusion_unseen_confidence_prob_global_blend_sw0p25_a0p50_hgm2r.json`
