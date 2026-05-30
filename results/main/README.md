# Main Results

This folder stores the curated runs intended for the thesis main tables.

Current primary selection:

- `shapenet/retrieval/rgb_zero_shot_both.json`
- `shapenet/retrieval/depth_zero_shot_both.json`
- `shapenet/retrieval/fusion_zero_shot_alpha0p50_both.json`

The safe PEFT retrieval results used in Table 5-4 are stored at the
`results/` root as `fusion_*_hgm2r.json` files.

Matching reported prompt checkpoints:

- `shapenet/prompt_tuning/coop_fusion_nctx8_random_seed0.json`
- `shapenet/prompt_tuning/coop_fusion_nctx8_random_seed0.pt`
- `shapenet/prompt_tuning/cocoop_fusion_nctx8_random_seed0_safe.json`
- `shapenet/prompt_tuning/cocoop_fusion_nctx8_random_seed0_safe.pt`

Raw duplicate retrieval outputs and earlier VL exploration results have been
removed from the stored project copy.
