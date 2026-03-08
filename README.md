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

# Run a single training experiment (~7 min including compile + eval)
uv run train.py

# Start autonomous research
# Point Claude Code (or any agent) at program.md and let it go
```

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
- **Smaller eval token budget.** Reduced for faster iteration (~52s eval vs ~11min on full budget). Same `evaluate_bpb` function from `prepare.py`.
- **~7 min experiment cycle.** 5 min training + ~11s compile + ~52s eval. Expect ~8-9 experiments/hour, ~70 overnight.
- **MFU reporting is placeholder.** No Apple Silicon FLOPs benchmark exists equivalent to H100_BF16_PEAK_FLOPS. `peak_vram_mb` reports MLX unified memory via
-
- # autoresearch-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops controlled entirely through `program.md`. This fork preserves every design rule — 5-minute wall-clock budget, single mutable `train.py`, one metric (`val_bpb`), keep/revert via git — and runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch or CUDA required.

## Quick start

Requirements: Apple Silicon Mac (M1/M2/M3/M4), Python 3.10+, uv.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py
uv run train.py
```

Then point Claude Code (or any agent) at `program.md` and let it go.

## Overnight Results

Three machines ran autonomously for 6-12 hours, each discovering its own optimal configuration.

| Machine | Optimizer | Experiments | Best val_bpb | Improvement |
|---|---|---|---|---|
| M4 Max 128GB | AdamW | ~50 | **1.295** | 19% |
| M4 Max 128GB (#2) | AdamW + surface gates | ~30 | 1.339 | 17% |
| Mac Mini | Muon + AdamW | 30 | 1.462 | 24% |

Upstream H100 reference: val_bpb 0.998 in the same 5-minute budget.

### What the Loop Found

Every machine converged on the same core insight: in a fixed 5-minute window, more optimizer steps beats more parameters.

- **DEPTH=4 over DEPTH=8.** Half the parameters, roughly 2x the training steps. All three machines converged here.
- **Smaller batch sizes.** TOTAL_BATCH_SIZE 2^14-2^13 consistently beat 2^17 by fitting more steps into the budget.
- **Lean MLP.** 3x expansion beat 4x. On the Mac Mini, 2x worked better.
- **Schedule tuning matters.** WARMDOWN_RATIO and FINAL_LR_FRAC were significant on all machines.

**Muon on Apple Silicon (first-ever tuning):**

- **NS_STEPS=3 beats NS_STEPS=5.** Fewer Newton-Schulz iterations means faster steps means more total steps.
- **Muon is hardware-dependent.** On the Mac Mini (constrained compute), Muon was the early breakthrough. On the M4 Max, plain AdamW with more steps still won. Muon makes each step count more — which matters most when you can't get many steps.

The most interesting result: the same loop, given different hardware, found genuinely different optimal configurations. That's exactly what autoresearch is designed to do.

## How it works

Three files that matter:

- **`prepare.py`** — data prep, tokenizer, dataloader, evaluation. Not modified.
- **`train.py`** — model, optimizer, training loop. The agent edits this. Includes optional Muon with tunable knobs.
- **`program.md`** — agent instructions. Point your agent here.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon, unified memory.
- **Muon included.** Set `USE_MUON = True` to enable. Recommended for memory-constrained hardware.
- **Smaller eval token budget.** ~52s eval for faster iteration.
- **~6-7 min experiment cycle.** 8-9 experiments/hour, 60-80 overnight.

## Recommended Defaults

Based on overnight results across three machines:

```python
DEPTH = 4
TOTAL_BATCH_SIZE = 2**14
MLP_EXPANSION = 3
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.1
USE_MUON = False          # True for constrained hardware
NS_STEPS = 3              # if using Muon
```

Starting points. The loop will find better settings for your hardware.


## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) — MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. Original copyright preserved. See [LICENSE](LICENSE).
