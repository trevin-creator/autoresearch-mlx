# autoresearch-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops controlled through `program.md`. This port keeps the same basic rules: one mutable `train.py`, one metric (`val_bpb`), a fixed 5-minute training budget, and keep-or-revert via git. It runs natively on Apple Silicon through [MLX](https://github.com/ml-explore/mlx), so there is no PyTorch or CUDA dependency.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# one-time data + tokenizer prep
uv run prepare.py

# run one 5-minute training experiment
uv run train.py
```

Then point Claude Code or another coding agent at `program.md` and let it run the loop.

## What matters

- `prepare.py` - data prep, tokenizer, dataloader, and evaluation. Treat as fixed.
- `train.py` - model, optimizer, and training loop. This is the file the agent edits.
- `program.md` - the autonomous experiment protocol.
- `results.tsv` - logged experiment history.

The loop is the same as upstream: edit `train.py`, run a fixed-budget experiment, read `val_bpb`, keep the change if it wins, revert if it loses, and repeat.

## Best results

### Current best: `1.280` val_bpb (M4 Max, March 2026)

The `autoresearch/mar19` branch achieved **1.280 val_bpb** starting from a 1.623 baseline — a 21% improvement over 20 kept experiments out of ~40 total. This is the best known public result for Apple Silicon.

| val_bpb | Description |
|---:|---|
| 1.623 | Baseline |
| 1.540 | Halve batch to 2^15 (more optimizer steps) |
| 1.402 | Halve batch to 2^14 |
| 1.394 | Full causal attention (remove sliding windows) |
| 1.388 | Value embeddings on all layers |
| 1.363 | **Peri-LN** (Gemma 2 / OLMo 2 post-sub-layer norm) |
| 1.344 | **Muon optimizer** (Newton-Schulz matrix sign for weight matrices) |
| 1.306 | Muon LR sweep to 0.01 |
| 1.300 | Rebalance embedding LR for Muon |
| 1.295 | Muon beta1=0.9 |
| 1.290 | **mx.fast.rms_norm** (optimized Metal kernel) |
| 1.287 | **Sparse attention gate** (nanoGPT speedrun Record 28) |
| 1.283 | **Nesterov momentum** in Muon |
| **1.280** | Warmup + warmdown tuning for Muon |

The biggest wins came from three categories: step-count maximization (batch reduction, 0.22 bpb), research-inspired optimizer changes (Muon + tuning, 0.11 bpb), and modern architecture techniques (Peri-LN + attention gate, 0.03 bpb).

### Comparison

| Machine | Best val_bpb | Notes |
|---|---:|---|
| **M4 Max (this run)** | **1.280** | Muon + Peri-LN + research-driven loop |
| M4 Max (prior best) | 1.295 | AdamW-only |
| M4 Max #2 | 1.331 | Different technique stack |
| Mac Mini | 1.353 | Long overnight run |
| H100 (Karpathy) | 0.970 | ~96x faster hardware, not comparable |

### What makes this run different

The key techniques that pushed past the prior 1.295 Apple Silicon best were all sourced from recent ML research:

- **Muon optimizer** (Keller Jordan, nanoGPT speedrun 2024) — replaces AdamW for weight matrices with Newton-Schulz orthogonalized momentum. The single biggest post-batch-reduction improvement.
- **Peri-LN** (Kim et al., ICML 2025; adopted by Gemma 2, OLMo 2) — adds normalization after each sub-layer output, not just before.
- **Sparse attention gate** (nanoGPT speedrun Record 28, Aug 2025) — per-head output gating via a small sigmoid on input features.
- **Nesterov Muon** — Nesterov lookahead momentum before Newton-Schulz orthogonalization.
- **mx.fast.rms_norm** — MLX's optimized Metal kernel for RMSNorm, replacing a multi-op inline implementation.

## Enhanced experiment protocol

This run used an enhanced version of the experiment loop (documented in `program.md`) that adds two systems on top of the basic keep-or-revert protocol:

### Literature-driven experiment selection

Rather than relying solely on intuition or grid search, the agent actively consults recent ML research (arXiv, ICML, NeurIPS, ICLR, nanoGPT speedrun community) to source experiment ideas. Research notes are saved to `literature/` for persistence across sessions. This directly produced the Muon, Peri-LN, sparse attention gate, and Nesterov momentum wins — all sourced from papers or community results published in 2024-2025.

The protocol includes targeted literature refreshes when the agent hits a plateau (3+ consecutive discards), which in this run immediately produced three consecutive wins (mx.fast.rms_norm, sparse attention gate, Nesterov Muon).

### Strategy knowledge base

The `strategy/` directory maintains four curated files that the agent reads before every experiment and updates after every experiment:

- **`strategy/learnings.md`** — Hardware- and config-specific insights with confidence levels. Not just "X failed" but "X failed *because* Y, which implies Z for future experiments." Example: "Any change adding per-step compute fails because the model is step-count-limited — except when the per-step quality improvement is large enough (Muon threshold: ~0.02 bpb per step)."

- **`strategy/hypotheses.md`** — Prioritized queue of untested ideas with theoretical rationale. New hypotheses are generated from literature findings and from analyzing why previous experiments succeeded or failed.

- **`strategy/near-misses.md`** — Experiments within noise margin (~0.01 bpb) that deserve revisiting after the config changes. Several ideas that failed at the original config later worked in a new context (e.g., warmdown tuning post-Muon).

- **`strategy/interactions.md`** — Known parameter couplings (e.g., "after switching to Muon, all learning rates need re-tuning"). Consulted before every experiment to avoid testing changes in isolation when they have known dependencies.

The pre-experiment checklist requires the agent to: (1) pick from the hypothesis queue, (2) check for parameter interactions, (3) scan near-misses for revisit candidates, (4) state a prediction and rationale, and (5) decide whether to test a single change or a deliberate multi-change bundle. This systematic approach reduced wasted experiments compared to naive hill climbing.

### Why this matters

The standard autoresearch loop is a greedy search — try one thing, keep or revert, repeat. This works but has known limitations: it can't discover synergistic combinations, it wastes experiments on low-probability ideas, and it doesn't learn from failures beyond "that didn't work."

The strategy knowledge base addresses these by maintaining a persistent model of *why* things work, not just *whether* they work. The literature integration ensures the agent draws on the broader ML research community rather than rediscovering known results from scratch. Together, these turned the experiment loop from a random walk into a directed search informed by both local evidence and external knowledge.

## Prior results

The initial hardware-local walk from the default baseline:

| Commit | val_bpb | Status | Description |
|---|---:|---|---|
| `383abb4` | 2.667000 | keep | baseline (AdamW, default config) |
| `909dd59` | 2.588904 | keep | halve total batch size to `2^16` |
| `4161af3` | 2.533728 | keep | increase matrix LR to `0.04` |
| `5efc7aa` | 1.807902 | keep | reduce depth from `8` to `4` |

That result already shows the core Apple Silicon pattern: with a fixed 5-minute wall clock, smaller faster-training models can beat larger ones simply by fitting more optimizer steps into the budget.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon training with unified memory.
- **Muon optimizer for weight matrices.** The `autoresearch/mar19` branch includes a full Muon implementation (Newton-Schulz matrix sign with Nesterov momentum), while keeping AdamW for embeddings and scalar parameters.
- **Peri-LN normalization.** Post-sub-layer norm in addition to pre-norm, following Gemma 2 / OLMo 2.
- **Strategy knowledge base.** The `strategy/` directory and enhanced `program.md` protocol add structured experiment selection, literature integration, and persistent cross-session learning.
- **Smaller eval token budget.** Reduced for faster iteration on Apple Silicon while keeping the same `evaluate_bpb` interface in `prepare.py`.
- **Roughly 6-7 minutes per experiment.** Expect 5 minutes of training plus compile and eval overhead.
- **MFU reporting is placeholder.** There is no Apple Silicon equivalent to the H100 FLOPs reference used upstream.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) - autoresearch and nanochat
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) - MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) - MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. See [LICENSE](LICENSE).
