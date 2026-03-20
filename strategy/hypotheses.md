# Hypotheses

Untested experiment ideas with theoretical rationale.

**Current best: 1.295 bpb. Strategy: research-inspired architectural changes over parameter tweaking.**

---

## Active — Awaiting Literature Refresh

These ideas are pending results from the current literature research agents. Will be prioritized after results arrive.

### H20: xIELU activation (arXiv:2411.13010)
- **Change**: Replace SqReLU with xIELU — a trainable piecewise activation with `x^2/2 + x` for positive inputs.
- **Rationale**: Beat both SwiGLU and SqReLU at 1.1B-3B scale. If it works at 15M scale, it's a drop-in replacement.
- **Status**: Awaiting implementation details from literature agent
- **Priority**: HIGH (pending research)

### H21: GLU + SqReLU (gated SqReLU)
- **Change**: `sqrelu(W1(x)) * W3(x)` — add gating to our working activation. Reduce hidden dim to compensate for extra params.
- **Rationale**: Gets gating benefits without changing the activation function. SwiGLU alone failed, but SwiGLU uses SiLU which we know is worse than SqReLU here.
- **Status**: Awaiting literature confirmation
- **Priority**: HIGH

### H22: Muon scheduling / momentum warmup
- **Change**: Start Muon beta1 lower (e.g., 0.85) and ramp to 0.95 over first ~300 steps.
- **Rationale**: Early training has noisier gradients, lower momentum helps. Later training benefits from more smoothing.
- **Status**: Awaiting research on Muon-specific schedules
- **Priority**: MEDIUM

### H23: Deep Delta Learning (arXiv:2601.00417)
- **Change**: Rank-1 perturbation of residual connections. Per-layer learned unit vector + scalar.
- **Rationale**: Very recent (Jan 2026), consistent improvement across scales. Minimal param overhead.
- **Status**: Awaiting full implementation details
- **Priority**: MEDIUM

### H24: SOAP optimizer (arXiv:2409.11321)
- **Change**: Alternative to Muon for weight matrices.
- **Status**: Awaiting comparison research
- **Priority**: LOW (Muon is already working well)

---

## Active — Ready to Test

### H25: HEAD_DIM=64 revisit (4 heads, current config)
- **Change**: HEAD_DIM 128→64
- **Rationale**: Previously failed at old config (1.695 vs 1.623, fewer steps). But now: Muon (+better per-step quality), Peri-LN (+stability), batch=2^14 (+more steps), VE on all layers (+capacity). The context has changed dramatically.
- **Risk**: Medium — head dim change affects attention compute, may still lose too many steps
- **Status**: Ready to test
- **Priority**: MEDIUM

### H26: Smaller wte init scale
- **Change**: wte init from `normal * 1.0` to `uniform(-scale, scale)` matching other weights
- **Rationale**: wte is initialized at much larger scale than other weights. The post-embedding norm clamps it, but gradient dynamics may differ.
- **Status**: Was committed but interrupted before running
- **Priority**: LOW

---

## Deprioritized

- H1: MATRIX_LR=0.03 (0.01 confirmed optimal for Muon)
- H3/H5: Cosine warmdown / warmup (1.409 as bundle, near-miss)
- H8: ADAM_BETAS tuning for non-Muon params (low expected impact)
- All other hyperparameter tweaks

---

## Tested / Resolved

See strategy/learnings.md for full results of all ~30 experiments.
