# Result Layout

Raw script outputs remain unchanged for compatibility:

- `results/unseen_retrieval/`: retrieval JSONs written directly by evaluation scripts
- `results/prompt_tuning/`: CoOp and CoCoOp checkpoints plus training summaries
- `results/lora/`: LoRA checkpoints and training summaries

Curated thesis-friendly copies are organized below:

- `results/main/shapenet/retrieval/`: main ShapeNet retrieval results used in the primary tables
- `results/main/shapenet/prompt_tuning/`: checkpoints and summaries for the reported CoOp and CoCoOp runs
- `results/ablations/shapenet/retrieval/`: ShapeNet retrieval ablations such as context length, seeds, and extra rerank variants
- `results/ablations/shapenet/prompt_tuning/`: prompt-learning ablations and extra prompt checkpoints
- `results/ablations/shapenet/lora/`: visual LoRA extension results
- `results/ablations/os_mn40_core/retrieval/`: OS-MN40-core extension-dataset runs
- `results/debug/backups/`: backup snapshots and non-canonical files kept for traceability

The curated folders contain copies only. Existing scripts still write to the raw output folders above.

Note:

- Some raw files in `results/unseen_retrieval/` reuse generic names across different datasets.
- When writing the thesis or preparing slides, prefer the curated copies under `results/main/` and `results/ablations/`.
