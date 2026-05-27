# CLIP-based Open-set 3D Object Retrieval

This project builds a ShapeNet-based open-set 3D object retrieval pipeline and studies a shared CLIP visual backbone for RGB views and point-cloud-derived depth maps.

Main pipeline:

- render `12` RGB views for each 3D object
- sample a `2048`-point point cloud for each object
- project the point cloud into `12` depth maps with rendered camera parameters
- extract pooled CLIP features for RGB views and depth maps
- evaluate unseen retrieval under a fixed `seen/unseen` protocol
- build an RGB+Depth Fusion visual retrieval baseline
- train PEFT modules only on seen classes
- evaluate safe Fusion+PEFT retrieval on unseen query-gallery splits without using unseen class names or unseen labels for ranking

## Main Protocol

- Total categories used: `50`
- Seen classes: `10`
- Unseen classes: `40`
- Seen split: `80% train_seen`, `20% val_seen`
- Unseen split: `70% gallery_unseen`, `30% query_unseen`

Fixed protocol file:

- `configs/splits/shapenet_seen10_unseen40_seed0.json`

## Safe Evaluation Rule

All current main PEFT results follow this rule:

- Training may use `train_seen` labels and validation may use `val_seen` labels.
- Test-time ranking must not use unseen class names, unseen labels, or gallery labels.
- Unseen labels are used only after ranking, for `mAP`, `NDCG@100`, `ANMRR`, and `Recall@100`.

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

### 4. Fusion + Adapter visual-only retrieval

- Base feature: `RGB+Depth Fusion`
- Adapter position: after object-level Fusion features
- Training: seen-class supervision on `train_seen`
- Training script: `scripts/adapter/train_fusion_adapter.py`
- Ranking: cosine similarity of post-fusion Adapter features
- Test-time text anchors: none
- Legacy `scripts/adapter/train_clip_adapter.py` adapts RGB and depth before fusion and is not used for this main result.

### 5. Fusion + LoRA visual-only retrieval

- Base feature: RGB and depth features re-extracted by a LoRA-adapted CLIP image encoder
- LoRA target: selected CLIP visual transformer blocks
- Training: seen-class supervision on `train_seen`
- Ranking: cosine similarity of LoRA-adapted Fusion features
- Test-time text anchors: none

### 6. Fusion + CoOp Seen-anchor retrieval

- Prompt tuning: shared CoOp context learned on `train_seen`
- Test-time text anchors: seen class names only
- Ranking: Fusion visual similarity plus seen-anchor response similarity
- Unseen class names are not used for ranking

### 7. Fusion + CoCoOp Seen-anchor retrieval

- Prompt tuning: conditional CoCoOp prompt learner trained on `train_seen`
- Test-time text anchors: seen class names only
- Ranking: Fusion visual similarity plus conditional seen-anchor response similarity
- Unseen class names are not used for ranking

## Main Metric Style

- Main tables in this project now use the `HGM2R-style` evaluator.
- Primary reported metrics are `mAP`, `NDCG@100`, `ANMRR`, and `Recall@100`.
- The previous `legacy` evaluator is still available in code for optional compatibility checks, but it is no longer the default reporting style.

## Current Main Results

Base visual retrieval:

| Method | Input | mAP | NDCG@100 | ANMRR | Recall@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| RGB | RGB multi-view | 0.5512 | 0.7582 | 0.4577 | 0.2294 |
| Depth | Point-cloud depth maps | 0.4762 | 0.6896 | 0.5284 | 0.1805 |
| RGB+Depth Fusion | RGB + Depth | 0.5701 | 0.7709 | 0.4418 | 0.2363 |

Safe Fusion+PEFT retrieval:

| Method | mAP | NDCG@100 | ANMRR | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| RGB+Depth Fusion | 0.5701 | 0.7709 | 0.4418 | 0.2363 |
| Fusion + Adapter | 0.5839 | 0.7763 | 0.4304 | 0.2412 |
| Fusion + LoRA | 0.6073 | 0.7914 | 0.4118 | 0.2497 |
| Fusion + CoOp Seen-anchor | 0.5904 | 0.7781 | 0.4251 | 0.2419 |
| Fusion + CoCoOp Seen-anchor | 0.5751 | 0.7736 | 0.4374 | 0.2382 |

Result files:

- `results/fusion_baseline_hgm2r.json`
- `results/fusion_adapter_visual_only_hgm2r.json`
- `results/fusion_lora_visual_only_hgm2r.json`
- `results/fusion_coop_seen_anchor_w0p25_hgm2r.json`
- `results/fusion_cocoop_seen_anchor_w0p05_bhat_p0p75_hgm2r.json`
- `results/peft_summary.csv`

Compared with the RGB+Depth Fusion baseline:

- `Fusion + Adapter`: `mAP +0.0138`, `NDCG@100 +0.0054`, `ANMRR -0.0114`
- `Fusion + LoRA`: `mAP +0.0372`, `NDCG@100 +0.0205`, `ANMRR -0.0301`
- `Fusion + CoOp Seen-anchor`: `mAP +0.0203`, `NDCG@100 +0.0072`, `ANMRR -0.0167`
- `Fusion + CoCoOp Seen-anchor`: `mAP +0.0049`, `NDCG@100 +0.0027`, `ANMRR -0.0044`

## Main Ablations

Fusion weight:

| alpha | Meaning | mAP | NDCG@100 | ANMRR |
| ---: | --- | ---: | ---: | ---: |
| 0.00 | Depth only | 0.4762 | 0.6896 | 0.5284 |
| 0.25 | More depth weight | 0.5430 | 0.7544 | 0.4667 |
| 0.50 | Balanced RGB and depth | 0.5701 | 0.7709 | 0.4418 |
| 0.75 | More RGB weight | 0.5554 | 0.7615 | 0.4544 |
| 1.00 | RGB only | 0.5512 | 0.7582 | 0.4577 |

CoCoOp Seen-anchor sensitivity:

| Setting | mAP | NDCG@100 | ANMRR |
| --- | ---: | ---: | ---: |
| `w=0.25, cosine` | 0.5492 | 0.7531 | 0.4642 |
| `w=0.03, Bhattacharyya, p=0.75` | 0.5732 | 0.7727 | 0.4391 |
| `w=0.05, Bhattacharyya, p=0.50` | 0.5724 | 0.7722 | 0.4398 |
| `w=0.05, Bhattacharyya, p=0.75` | 0.5751 | 0.7736 | 0.4374 |

Post-fusion Adapter hidden dimension:

| hidden_dim | mAP | NDCG@100 | ANMRR |
| ---: | ---: | ---: | ---: |
| 64 | 0.5838 | 0.7761 | 0.4303 |
| 128 | 0.5839 | 0.7763 | 0.4304 |
| 256 | 0.5819 | 0.7755 | 0.4324 |

## Project Layout

- `tools/`: preprocessing utilities, including multi-view rendering, point-cloud sampling, and point-cloud-to-depth projection.
- `scripts/`: experiment entry points grouped into `scripts/main`, `scripts/prompt`, `scripts/adapter`, `scripts/os_mn40_core`, and `scripts/lora`.
- `models/`: lightweight CLIP wrappers plus CoOp/CoCoOp prompt learner modules and the residual feature Adapter.
- `utils/`: shared helpers for CLIP loading, feature loading, protocol handling, retrieval metrics, and safety metadata.
- `configs/`: default paths and saved protocol splits.
- `results/`: experiment outputs. Direct script outputs currently go to `results/unseen_retrieval`, `results/prompt_tuning`, `results/adapter`, and `results/lora`.
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
- LoRA RGB features: `D:/1Ahaha/AA3d/output_224_clip_feat_lora_r8`
- LoRA Depth features: `D:/1Ahaha/AA3d/output_feat_depth_maps_lora_r8`

## Result Organization

Current script outputs remain unchanged for compatibility:

- `results/unseen_retrieval/`: retrieval JSON outputs written by evaluation scripts
- `results/`: safe Fusion+PEFT JSON summaries from `run_fusion_peft_retrieval.py`
- `results/prompt_tuning/`: CoOp and CoCoOp checkpoints plus training summaries
- `results/adapter/`: Adapter checkpoints plus training summaries
- `results/lora/`: LoRA checkpoints plus training summaries

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

Validate the safe Fusion baseline with the new PEFT evaluator:

```bash
python scripts/main/run_fusion_peft_retrieval.py --method fusion --metric_style hgm2r --output_dir results --save_name fusion_baseline_hgm2r.json
```

Train a post-fusion Adapter on seen classes:

```bash
python scripts/adapter/train_fusion_adapter.py --epochs 20 --hidden_dim 128 --save_name fusion_post_adapter_h128_seed0.pt
```

Run safe Adapter visual-only retrieval:

```bash
python scripts/main/run_fusion_peft_retrieval.py --method adapter --adapter_ckpt results/adapter/fusion_post_adapter_h128_seed0.pt --metric_style hgm2r --output_dir results --save_name fusion_adapter_visual_only_hgm2r.json
```

Train a shared CoOp prompt on seen classes:

```bash
python scripts/prompt/train_prompt_coop.py --mode fusion --epochs 20 --n_ctx 8 --ctx_init " " --save_name coop_fusion_nctx8_random_seed0.pt
```

Run safe CoOp Seen-anchor retrieval:

```bash
python scripts/main/run_fusion_peft_retrieval.py --method coop_seen_anchor --prompt_ckpt results/prompt_tuning/coop_fusion_nctx8_random_seed0.pt --seen_anchor_weight 0.25 --metric_style hgm2r --output_dir results --save_name fusion_coop_seen_anchor_w0p25_hgm2r.json
```

Train a CoCoOp prompt on seen classes:

```bash
python scripts/prompt/train_prompt_cocoop.py --mode fusion --epochs 20 --n_ctx 8 --ctx_init " " --meta_hidden_dim 64 --batch_size 16 --prompt_chunk_size 64 --save_name cocoop_fusion_nctx8_random_seed0_safe.pt
```

Run safe CoCoOp Seen-anchor retrieval:

```bash
python scripts/main/run_fusion_peft_retrieval.py --method cocoop_seen_anchor --prompt_ckpt results/prompt_tuning/cocoop_fusion_nctx8_random_seed0_safe.pt --seen_anchor_weight 0.05 --seen_anchor_similarity bhattacharyya --seen_anchor_power 0.75 --prompt_batch_size 4 --prompt_chunk_size 8 --metric_style hgm2r --output_dir results --save_name fusion_cocoop_seen_anchor_w0p05_bhat_p0p75_hgm2r.json
```

Train a visual LoRA branch on CLIP and save the checkpoint:

```bash
python scripts/lora/train_clip_lora.py --mode fusion --epochs 5 --rank 8 --save_name clip_lora_fusion_r8_seed0.pt
```

Extract LoRA-adapted RGB and depth features:

```bash
python scripts/lora/extract_clip_features_lora.py --lora_ckpt results/lora/clip_lora_fusion_r8_seed0.pt --modality rgb --output_root D:/1Ahaha/AA3d/output_224_clip_feat_lora_r8
python scripts/lora/extract_clip_features_lora.py --lora_ckpt results/lora/clip_lora_fusion_r8_seed0.pt --modality depth --output_root D:/1Ahaha/AA3d/output_feat_depth_maps_lora_r8
```

Run safe LoRA visual-only retrieval:

```bash
python scripts/main/run_fusion_peft_retrieval.py --method lora --lora_rgb_feat_root D:/1Ahaha/AA3d/output_224_clip_feat_lora_r8 --lora_depth_feat_root D:/1Ahaha/AA3d/output_feat_depth_maps_lora_r8 --metric_style hgm2r --output_dir results --save_name fusion_lora_visual_only_hgm2r.json
```

Summarize safe PEFT JSON files:

```bash
python scripts/main/summarize_fusion_peft_results.py --result_dir results --output results/peft_summary.csv --strict_safety
```

Legacy note: `scripts/main/run_vl_retrieval.py` is kept only for compatibility with earlier exploratory experiments and is not used for the safe main results above.

If you need both the main `HGM2R-style` metrics and the older compatibility metrics in the same file, use:

```bash
python scripts/main/run_unseen_retrieval.py --mode fusion --metric_style both
python scripts/main/run_fusion_peft_retrieval.py --method fusion --metric_style both --output_dir results
```

Key outputs:

- `results/unseen_retrieval/rgb_zero_shot_hgm2r.json` or `rgb_zero_shot_both.json`
- `results/unseen_retrieval/depth_zero_shot_hgm2r.json` or `depth_zero_shot_both.json`
- `results/unseen_retrieval/fusion_zero_shot_alpha0p50_hgm2r.json` or `fusion_zero_shot_alpha0p50_both.json`
- `results/fusion_baseline_hgm2r.json`
- `results/adapter/fusion_post_adapter_h128_seed0.pt` and `fusion_post_adapter_h128_seed0.json`
- `results/fusion_adapter_visual_only_hgm2r.json`
- `results/lora/clip_lora_fusion_r8_seed0.pt` and `clip_lora_fusion_r8_seed0.json`
- `results/fusion_lora_visual_only_hgm2r.json`
- `results/fusion_coop_seen_anchor_w0p25_hgm2r.json`
- `results/fusion_cocoop_seen_anchor_w0p05_bhat_p0p75_hgm2r.json`
- `results/peft_summary.csv`
