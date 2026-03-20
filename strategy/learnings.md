# Learnings

Hardware- and config-specific insights derived from experiments on this machine.
Updated after every experiment. Each entry explains *why* something worked or didn't, not just whether it did.

**Current best: 1.295 bpb** (13 kept experiments from 1.623 baseline)

---

## Core Principles (derived from ~30 experiments)

### Step count vs per-step quality trade-off (high confidence)

The model operates in a **step-count-dominated regime** at batch=2^14 (~1500 steps in 5 min). Any change that adds per-step compute costs measurable steps. Small additions (z-loss, EMA, embed shortcircuit) consistently fail because the step loss outweighs tiny quality gains.

**Exception**: Muon optimizer added significant per-step compute (Newton-Schulz iterations, ~100 fewer steps) but its per-step quality improvement was large enough to overcome the step penalty. The threshold appears to be ~0.02+ bpb improvement per step to justify the compute cost.

### Muon optimizer with tuned hyperparams (high confidence)

Muon for weight matrices is the single best change after batch reduction. Key findings:
- **LR**: 0.04→1.344, 0.02→1.318, **0.01→1.306**, 0.005→1.311. Optimum at 0.01 (4x lower than Adam's optimal 0.04).
- **Beta1**: **0.9→1.295**, 0.85→1.296, 0.95→1.306. Less momentum (0.9) is better with noisy small-batch gradients.
- **Weight decay**: WD=0.2 is better than WD=0.0 for Muon matrices. WD provides useful regularization that Muon's matrix sign doesn't replace.
- **Scope**: Muon for block weights only. Applying Muon to lm_head was much worse (1.410) — embeddings and output heads should stay on Adam.
- **Newton-Schulz iterations**: 5 is better than 3 (1.308 vs 1.306). The quality of the matrix sign approximation matters.

### Embedding LR needs rebalancing after Muon (high confidence)

EMBEDDING_LR 0.6→0.3 gave 1.300→ improvement after Muon was introduced. Muon makes weight matrices learn faster, so embedding LR needed to come down to avoid embeddings over-learning relative to matrices. Always re-tune adjacent LRs after a major optimizer change.

---

## Architecture Findings

### Full attention beats sliding window at 4 layers (high confidence)

WINDOW_PATTERN "SSSL"→"LLLL" gave 1.394 vs 1.402 with MORE steps. At 4 layers, restricting context is too aggressive. Uniform masks also cache better.

### VE on all layers is optimal (high confidence)

VE on all 4 layers (vs alternating 2) gave 1.388 vs 1.394. VE is the cheapest way to add capacity (embedding lookups, not matmuls). Total VE is now ~8.4M of 15.7M params (53%).

### Peri-LN is the biggest single architectural win (high confidence)

Post-sub-layer norm (`x + norm(attn(norm(x)))`) gave 1.363 vs 1.387 — 0.024 improvement. The model needs both pre-norm AND post-norm. Asymmetric peri-LN (only post-attn, not post-MLP) is slightly worse (1.301 vs 1.300). Both post-norms matter.

### Fixed RMSNorm is better than learnable or DyT (high confidence)

Learnable RMSNorm: 1.433 vs 1.402. DyT: 1.506 vs 1.363. The fixed `norm(x) = x * rsqrt(mean(x^2) + eps)` is the right choice. Learnable scale params add nothing useful; DyT's tanh saturation is fundamentally wrong for this architecture.

### Logit softcap = 15 is correct (high confidence)

Removing cap: 1.430 (more steps but worse quality). Cap=30: 1.370 vs cap=15 at 1.363. The tight cap stabilizes gradients. Don't change.

### SqReLU is better than SiLU (high confidence, current config)

SiLU gave 1.325 vs SqReLU at 1.300. SqReLU's sparsity (hard zero + squaring) is more efficient. However, GLU variants (gating) haven't been tested with SqReLU — worth exploring.

### Depth increases are too expensive (high confidence)

DEPTH=5: 1.336 (779 steps, 30.9M params). DEPTH=6: 1.593 (705 steps). Each extra layer adds ~15M params (mostly VE) which halves step count. Even Muon can't compensate.

### Width increases are too expensive (high confidence)

dim=384 (ASPECT_RATIO=96): 1.478 (1035 steps, 26M params). MLP/attention compute grows quadratically with dim.

### MLP ratio changes don't help (medium confidence)

3x MLP: 1.455 (less capacity). 5x MLP: 1.307 (more capacity but too expensive). 4x is the sweet spot.

---

## Optimizer Findings (non-Muon)

### ADAM_BETAS, WEIGHT_DECAY, warmdown are all tuned (high confidence)

- ADAM_BETAS=(0.8, 0.95) for non-Muon params: not tuned post-Muon, but the embedding/scalar params are a smaller fraction now
- WEIGHT_DECAY=0.2: WD=0.1 was worse (1.418)
- WARMDOWN_RATIO=0.3: 0.5 was worse, 0.2 was worse. 0.3 is the sweet spot
- FINAL_LR_FRAC=0.0: 0.05 was worse (1.410)
- Cosine warmdown + warmup(0.02): 1.409 — near-miss but not better than linear

### EMA of weights is harmful (medium confidence)

EMA with decay 0.99 and 0.995 both worse than raw weights. The linear warmdown already produces a well-converged final checkpoint.

---

## Dead Ends

- Gradient centralization: 1.692 (conflicts with existing normalization)
- Cautious Adam: 1.846 (rescaling too aggressive)
- Tied embeddings: 4.084 (LR conflict between input/output roles)
- Remove x0 skip connections: 1.527 (critical for training dynamics, even though 2121 steps)
- Remove post-embedding norm: 1.345
- Zero-init lm_head: 1.302 (near-zero 0.001 init is better than true zero)
- Embedding weight decay: 1.305 (embeddings need full freedom)

---

## Single-machine constraint (hard constraint)

All experiments run sequentially on one M4 Max. ~7 min per experiment. Strategy knowledge base compensates by making smarter picks.
