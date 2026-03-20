# Hypotheses

Untested experiment ideas with theoretical rationale. Includes both single changes and multi-change bundles.
Each entry tracks: the hypothesis, expected mechanism, dependencies, and status.

**Strategy note**: Hyperparameter tweaking (LR, WD, warmdown, betas) has reached diminishing returns. The current config is well-optimized for its architecture. Future experiments should focus on **architectural changes** and **research-inspired techniques** that change learning dynamics, not parameter grid search.

---

## Active Hypotheses — Architectural / Research-Inspired

### H11: EMA of weights for evaluation
- **Change**: Maintain an exponential moving average of model weights (decay ~0.995). Before evaluation, swap in EMA weights. This doesn't change training at all — only the checkpoint used for final eval.
- **Rationale**: With 1600 noisy small-batch updates, the final weights are a single noisy point. EMA smooths out the noise for a better eval checkpoint. This is standard practice (Polyak averaging) and essentially free improvement.
- **Risk**: Very low. Small memory overhead (~2x param storage). No training time cost beyond a few multiplies per step.
- **Status**: Untested
- **Priority**: HIGH

### H12: Tied input/output embeddings
- **Change**: Share weights between `wte` (input embedding) and `lm_head` (output projection). Since vocab_size=8192 and model_dim=256, both are [8192, 256] matrices. Tying them forces consistency and halves embedding param count.
- **Rationale**: Standard in many modern LLMs (T5, LLaMA at some scales). Reduces param count but forces the embedding to be useful for both input representation and output prediction. The freed parameters could be reinvested (e.g., slightly wider model).
- **Risk**: Medium — could hurt if the model benefits from separate embedding spaces.
- **Status**: Untested
- **Priority**: HIGH

### H13: Z-loss regularization
- **Change**: Add a small penalty on log(Z) where Z = sum(exp(logits)). Implementation: `z_loss = 1e-4 * mean(log(sum(exp(logits)))^2)`.
- **Rationale**: From PaLM and nanoGPT speedrun. Prevents logit drift and keeps the softmax numerically stable. Works synergistically with the existing logit capping (which we've confirmed is valuable).
- **Risk**: Very low — tiny regularization term.
- **Status**: Untested
- **Priority**: HIGH

### H14: Embed shortcircuit (skip connection from embeddings to output)
- **Change**: Before the final lm_head projection, mix in the original embeddings: `x = x + alpha * x0` where x0 is the post-embedding representation and alpha is small (e.g., 0.1).
- **Rationale**: From nanoGPT speedrun. Creates a direct gradient path from the loss to the embeddings, improving embedding learning speed. Especially valuable for short training runs where embeddings may be undertrained.
- **Risk**: Low — small architectural change, easy to tune alpha.
- **Status**: Untested
- **Priority**: HIGH

### H15: Learnable RMSNorm (replace inline norm function)
- **Change**: Replace the fixed `norm(x)` with `nn.RMSNorm(n_embd)` which adds a learnable per-channel scale parameter.
- **Rationale**: Gives the model per-channel control over feature magnitudes. Tiny param increase (~256 params per usage). Standard in modern architectures.
- **Risk**: Very low.
- **Status**: Untested
- **Priority**: MEDIUM

### H16: Attention temperature scaling
- **Change**: Replace fixed `scale = 1/sqrt(head_dim)` with a learnable per-head temperature or use a different fixed scale (e.g., `1/sqrt(head_dim * 0.5)` for sharper attention).
- **Rationale**: The nanoGPT speedrun and other work found that sharper attention (higher effective temperature) can improve learning at small scales. With only 2 heads, each head needs to attend precisely.
- **Risk**: Low.
- **Status**: Untested
- **Priority**: MEDIUM

---

## Deprioritized — Hyperparameter Tweaks

These have reached diminishing returns. Only revisit if architectural changes shift the landscape.

- **H1**: MATRIX_LR=0.03 — 0.04 is near-optimal (0.02 and 0.06 both worse)
- **H3**: Cosine warmdown — tested as bundle with warmup, result 1.409 (near-miss)
- **H5**: Small warmup — tested as bundle with cosine, result 1.409 (near-miss)
- **H8**: ADAM_BETAS tuning — deprioritized per strategy shift

---

## Tested / Resolved

- **H2**: WD=0.1 → 1.418 (worse). WD=0.2 confirmed optimal.
- **H4**: Remove logit cap → 1.430 (worse despite more steps). Cap is beneficial.
- **H6**: DEPTH=6 → 1.593 (way worse, only 705 steps at 26.3M params)
- **H9**: Remove VE → 1.543 (way worse, lost 36% of params)
- **H10**: WARMDOWN_RATIO=0.2 → 1.429 (worse). 0.3 is optimal.
- MLP 3x ratio → 1.455 (worse, lost capacity)
- SwiGLU at dim=256 → 1.665 (worse, tested at old config)
- HEAD_DIM=64 → 1.695 (worse, tested at old config)
- Cautious Adam → 1.846 (way worse)
- Gradient centralization → 1.692 (worse)
