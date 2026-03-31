# CLIP-based Open-set 3D Object Retrieval

This project builds a ShapeNet-based open-set 3D object retrieval pipeline and studies a shared CLIP visual backbone for RGB views and point-cloud-derived depth maps.

Main pipeline:

- render `12` RGB views for each 3D object
- sample a `2048`-point point cloud for each object
- project the point cloud into `12` depth maps with rendered camera parameters
- extract pooled CLIP features for RGB views and depth maps
- evaluate unseen retrieval under a fixed `seen/unseen` protocol
- run zero-training visual-language retrieval on top of cached RGB-depth features
- optionally train CoOp or CoCoOp prompts on seen classes and reuse them for unseen retrieval

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

### 5. Visual-Language Retrieval + CoOp

- Visual branch: cached RGB+Depth fusion CLIP features
- Prompt tuning: shared CoOp prompt learned on `train_seen`
- Evaluation: `query_unseen -> gallery_unseen`
- Metrics: `mAP`, `NDCG@100`, `ANMRR`, `Recall@100`

### 6. Visual-Language Retrieval + CoCoOp

- Visual branch: cached RGB+Depth fusion CLIP features
- Prompt tuning: conditional CoCoOp prompt learned on `train_seen`
- Evaluation: `query_unseen -> gallery_unseen`
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
| VL Retrieval (fixed prompt) | 0.6073 | 0.7837 | 0.4086 | 0.2476 |
| VL Retrieval + CoOp | 0.6376 | 0.7941 | 0.3809 | 0.2554 |
| VL Retrieval + CoCoOp | 0.6202 | 0.7877 | 0.3970 | 0.2495 |

Result files:

- `results/unseen_retrieval/rgb_zero_shot_both.json`
- `results/unseen_retrieval/depth_zero_shot_both.json`
- `results/unseen_retrieval/fusion_zero_shot_alpha0p50_both.json`
- `results/unseen_retrieval/vl_fusion_unseen_confidence_prob_global_blend_sw0p25_a0p50_hgm2r.json`
- `results/unseen_retrieval/vl_fusion_coop_ctx8_random_hgm2r.json`
- `results/unseen_retrieval/vl_fusion_cocoop_ctx8_random_seed0_safe_hgm2r.json`

Compared with the zero-shot RGB+Depth baseline, VL Retrieval improves:

- `mAP`: `+0.0371`
- `NDCG@100`: `+0.0128`
- `ANMRR`: `-0.0332`
- `Recall@100`: `+0.0113`

Compared with fixed-prompt VL Retrieval, prompt tuning improves:

- `CoOp`: `mAP +0.0303`, `NDCG@100 +0.0104`, `ANMRR -0.0278`, `Recall@100 +0.0079`
- `CoCoOp`: `mAP +0.0130`, `NDCG@100 +0.0040`, `ANMRR -0.0117`, `Recall@100 +0.0020`

## Project Layout

- `tools/`: preprocessing utilities, including multi-view rendering, point-cloud sampling, and point-cloud-to-depth projection.
- `scripts/`: experiment entry points grouped into `scripts/main`, `scripts/prompt`, `scripts/os_mn40_core`, and `scripts/lora`.
- `models/`: lightweight CLIP wrappers plus CoOp/CoCoOp prompt learner modules.
- `utils/`: shared helpers for CLIP loading, protocol handling, retrieval metrics, and semantic reranking.
- `configs/`: default paths and saved protocol splits.
- `results/`: experiment outputs. Direct script outputs currently go to `results/unseen_retrieval` and `results/prompt_tuning`.
- `datasets/`: reserved for dataset notes, manifests, and benchmark adapters. Raw datasets are not stored in this repository.
- `CLIP/`: vendored OpenAI CLIP source used by this project.

## External Data Layout

This repository mainly stores code. Large datasets and intermediate artifacts are expected to live outside the repo under the current `D:/1Ahaha/AA3d` workspace.

Current external roots used by the project:

- ShapeNet raw models: `D:/1Ahaha/AA3d/ShapeNet`
- Rendered RGB views: `D:/1Ahaha/AA3d/output_224`
- Sampled point clouds: `D:/1Ahaha/AA3d/ShapeNet_PointClouds`
- Point-cloud-derived depth maps: `D:/1Ahaha/AA3d/depth_maps`
- RGB CLIP features: `D:/1Ahaha/AA3d/output_224_clip_feat`
- Depth CLIP features: `D:/1Ahaha/AA3d/output_feat_depth_maps`

## Result Organization

Current script outputs remain unchanged for compatibility:

- `results/unseen_retrieval/`: retrieval JSON outputs written by evaluation scripts
- `results/prompt_tuning/`: CoOp and CoCoOp checkpoints plus training summaries

Reserved folders for manual curation:

- `results/main/`: final tables or selected runs for the thesis
- `results/ablations/`: controlled comparison runs
- `results/debug/`: temporary or throwaway analysis outputs

## Run

Prepare data:

```bash
python tools/batch_render.py
python tools/batch_sample_pc_custom.py
python tools/pc_to_depth.py
```

Extract CLIP features:

```bash
python scripts/main/extract_rgb_features.py
python scripts/main/extract_depth_features.py
```

Build the fixed protocol:

```bash
python scripts/main/build_seen_unseen_protocol.py
```

Run zero-shot unseen retrieval:

```bash
python scripts/main/run_unseen_retrieval.py --mode rgb --metric_style hgm2r
python scripts/main/run_unseen_retrieval.py --mode depth --metric_style hgm2r
python scripts/main/run_unseen_retrieval.py --mode fusion --metric_style hgm2r
```

Run zero-training visual-language retrieval:

```bash
python scripts/main/run_vl_retrieval.py --mode fusion --metric_style hgm2r
```

Train a first-stage shared CoOp prompt on seen classes:

```bash
python scripts/prompt/train_prompt_coop.py --mode fusion --epochs 20 --n_ctx 8 --ctx_init " " --save_name coop_fusion_nctx8_random_seed0.pt
```

Run visual-language retrieval with the learned CoOp prompt:

```bash
python scripts/main/run_vl_retrieval.py --mode fusion --prompt_mode coop --prompt_ckpt results/prompt_tuning/coop_fusion_nctx8_random_seed0.pt --metric_style hgm2r --save_name vl_fusion_coop_ctx8_random_hgm2r.json
```

Train a first-stage CoCoOp prompt with cached fusion features:

```bash
python scripts/prompt/train_prompt_cocoop.py --mode fusion --epochs 20 --n_ctx 8 --ctx_init " " --meta_hidden_dim 64 --batch_size 16 --prompt_chunk_size 64 --save_name cocoop_fusion_nctx8_random_seed0_safe.pt
```

Run visual-language retrieval with the learned CoCoOp prompt:

```bash
python scripts/main/run_vl_retrieval.py --mode fusion --prompt_mode cocoop --prompt_ckpt results/prompt_tuning/cocoop_fusion_nctx8_random_seed0_safe.pt --prompt_batch_size 4 --prompt_chunk_size 8 --metric_style hgm2r --save_name vl_fusion_cocoop_ctx8_random_seed0_safe_hgm2r.json
```

If you need both the main `HGM2R-style` metrics and the older compatibility metrics in the same file, use:

```bash
python scripts/main/run_unseen_retrieval.py --mode fusion --metric_style both
python scripts/main/run_vl_retrieval.py --mode fusion --metric_style both
```

Key outputs:

- `results/unseen_retrieval/rgb_zero_shot_hgm2r.json` or `rgb_zero_shot_both.json`
- `results/unseen_retrieval/depth_zero_shot_hgm2r.json` or `depth_zero_shot_both.json`
- `results/unseen_retrieval/fusion_zero_shot_alpha0p50_hgm2r.json` or `fusion_zero_shot_alpha0p50_both.json`
- `results/unseen_retrieval/vl_fusion_unseen_confidence_prob_global_blend_sw0p25_a0p50_hgm2r.json`
- `results/unseen_retrieval/vl_fusion_coop_ctx8_random_hgm2r.json`
- `results/unseen_retrieval/vl_fusion_cocoop_ctx8_random_seed0_safe_hgm2r.json`
