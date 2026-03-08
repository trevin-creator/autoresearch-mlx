# autoresearch-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops controlled entirely through `program.md`. This fork preserves every design rule — 5-minute wall-clock budget, single mutable `train.py`, one metric (`val_bpb`), keep/revert via git — and runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch or CUDA required.

## Quick start

Requirements: Apple Silicon Mac (M1/M2/M3/M4), Python 3.10+, uv.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (one-time)
uv run prepare.py

# Run a single training experiment
uv run train.py

# Start autonomous research
# Point Claude Code (or any agent) at program.md and let it go
```

The checked-in defaults aim to fit 16 GB Apple Silicon Macs by keeping `FINAL_EVAL_BATCH_SIZE` conservative. If you have more unified memory and want faster evaluation, raise that value in `train.py`.

## How it works

Same as the original. Three files that matter:

- **`prepare.py`** — data prep, tokenizer, dataloader, evaluation. Not modified.
- **`train.py`** — model, optimizer, training loop. The agent edits this.
- **`program.md`** — agent instructions. Point your agent here.

The agent reads `program.md`, modifies `train.py`, runs a 5-minute experiment, checks `val_bpb`, and commits or reverts. Repeat overnight. Wake up to results.

## Results on M1 Mac Studio (48GB)

Starting from the upstream default configuration and running the autoresearch loop:

| Experiment | Change | val_bpb | Action |
|---|---|---|---|
| baseline | default config | 2.667 | keep |
| 1 | halve batch size | 2.589 | keep |
| 2 | 10x matrix LR | 2.534 | keep |
| 3 | depth 8 → 4 | 1.808 | keep |

Key finding: Apple Silicon throughput in a 5-minute window favors smaller, faster-training models. The autoresearch loop discovered this automatically — more optimizer steps beat more parameters when compute time is fixed.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon, unified memory.
- **AdamW only.** Muon optimizer port is future work.
- **Smaller eval token budget.** Reduced for faster iteration, while still using the same `evaluate_bpb` function from `prepare.py`.
- **16 GB-safe default eval batch.** The checked-in `FINAL_EVAL_BATCH_SIZE` is conservative enough for lower-memory Apple Silicon Macs; larger-memory machines can increase it for faster evaluation.
- **MFU reporting is placeholder.** No Apple Silicon FLOPs benchmark exists equivalent to H100_BF16_PEAK_FLOPS. `peak_vram_mb` reports MLX unified memory via `mx.metal.get_peak_memory()`.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) — MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. Original copyright preserved. See [LICENSE](LICENSE).
