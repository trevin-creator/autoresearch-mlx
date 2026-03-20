# Hypotheses

**Current best: 1.295 bpb.**

Prioritized by expected impact based on literature refresh (Mar 19 evening).

---

## Tier 1: HIGH expected impact

### H30: mx.fast.rms_norm replacing inline norm()
- **Change**: Replace `norm(x) = x * rsqrt(mean(x*x) + eps)` with `mx.fast.rms_norm(x, weight=None, eps=1e-5)`
- **Rationale**: Single optimized Metal kernel vs multiple element-wise ops. Called 20+ times per forward pass. Pure speed gain, no quality change.
- **Risk**: Very low — same math, faster kernel
- **Priority**: HIGHEST (compute-neutral speed win)

### H31: mx.compile the training step
- **Change**: Wrap forward+backward+optimizer into `mx.compile` decorated function
- **Rationale**: Fuses all element-wise ops into single graph. MLX docs show significant speedups. More steps in 5 min = better val_bpb.
- **Risk**: Medium — requires restructuring loop, compiled fns must be pure (no mx.eval inside)
- **Priority**: HIGH (but complex implementation)
- **Note**: May conflict with Muon's Newton-Schulz iterations which use loops. Test carefully.

### H32: xIELU activation (non-GLU, drop-in for SqReLU)
- **Change**: Replace `max(0,x)^2` with `x * sigmoid(beta*x) + alpha * min(0,x)^2`
- **Rationale**: Trainable activation that learns optimal shape per channel. Beat SqReLU 1-3% at small scale. 2 extra params per hidden dim (negligible). No GLU restructuring needed.
- **Risk**: Low — drop-in replacement, easy to revert
- **Priority**: HIGH

### H33: Warm-start Newton-Schulz
- **Change**: Cache previous orthogonalized matrix, use as starting point. Reduce iterations from 5 to 3.
- **Rationale**: Consecutive momentum matrices are similar. Warm start converges in fewer iterations. Recovers ~30-40% of Muon overhead → more steps.
- **Risk**: Low — if approximation quality drops, we just revert
- **Priority**: HIGH

---

## Tier 2: MEDIUM expected impact

### H34: GLU + SqReLU (already committed, pending run)
- **Change**: `sqrelu(W1(x)) * W3(x)` with 8/3x hidden dim for param parity
- **Rationale**: Adds multiplicative gating while keeping our best activation. Currently committed but was interrupted before running.
- **Priority**: MEDIUM (already ready to test)

### H35: Nesterov momentum in Muon
- **Change**: Compute Nesterov lookahead gradient before accumulating into momentum
- **Rationale**: Small but consistent ~0.5-1% convergence improvement per speedrun community
- **Risk**: Very low
- **Priority**: MEDIUM

### H36: Sparse Attention Gate (nanoGPT Record 28)
- **Change**: Sigmoid gate on first 12 dims of input modulates each head's output
- **Rationale**: Reduces attention sink behavior. ~24 extra params (12 * 2 heads).
- **Risk**: Low
- **Priority**: MEDIUM

### H37: SiLU^2 activation
- **Change**: Replace `max(0,x)^2` with `silu(x)^2`
- **Rationale**: SqReLU sharpness + SiLU smoothness. Zero extra params. SiLU alone failed (1.325) but squaring it may give the best of both.
- **Risk**: Low
- **Priority**: MEDIUM

---

## Tier 3: LOWER expected impact / higher risk

### H38: Cautious Muon (no rescaling)
- **Change**: Mask Muon updates where current and previous updates disagree in sign, WITHOUT rescaling
- **Rationale**: C-Adam with rescaling was terrible (1.846), but the problem was the rescaling, not the masking idea itself
- **Priority**: LOW

### H39: Multi-token prediction auxiliary head
- **Priority**: LOW (complex, uncertain payoff)

---

## Tested / Resolved
See strategy/learnings.md for all ~30 experiment results.
