# Micro-Architecture Research — March 19, 2026

Second targeted literature search, focused on pushing past 1.295 bpb at 15.7M params.

---

## Attention Head Initialization

- HEAD_DIM=64 (4 heads) was previously tested at old config and failed (1.695 vs 1.623). But that was pre-Muon, pre-Peri-LN, pre-batch-reduction. The context has changed enough that it's worth revisiting.
- No strong evidence for head-specific init when QK-norm is present (normalizes away init differences).

## Gradient Accumulation

- Current regime (batch=2^14, grad_accum=1) is already optimal for short-budget training.
- Research confirms smaller batches with more steps outperform larger batches for short runs.

## Residual Connection Scaling

- **Deep Delta Learning (DDL)** (arXiv:2601.00417, Jan 2026): Rank-1 perturbation of identity residual. Per-layer learned unit vector + scalar gate. Consistent improvement across scales but moderate implementation complexity.
- **DeepSeek Hyper-Connections**: Too complex for 4 layers.
- Our resid_lambdas + x0_lambdas + zero-init c_proj already implements a superset of ReZero.

## Token Mixing

- Hyena/RWKV not beneficial at seq_len=2048 — attention is competitive or faster.
- Sliding window warmup (SSSS → LLLL mid-training) is interesting but uncertain payoff.
- Embed shortcircuit (logit bypass) via `wte(idx) @ lm_head.weight.T` creates a direct prediction shortcut. Worth testing but adds a large matmul.

## MLP Modifications

- **SwiGLU**: Already tested at old config and failed (1.665). But that was pre-everything. Worth revisiting as a param-matched swap.
- **xIELU** (arXiv:2411.13010): Trainable piecewise activation. `x^2/2 + x` for positive, ELU-like for negative. Beat SwiGLU and SqReLU at 1.1B-3B scale.
- **GLU + SqReLU**: `sqrelu(W1(x)) * W3(x)` — gating with our working activation. Compromise between SwiGLU and current MLP.
- **Trivial MoE**: 2 experts at this scale probably not worth the router overhead.

## Summary: Top Candidates

1. xIELU activation (if research confirms small-scale applicability)
2. GLU + SqReLU (safer bet — keeps working activation, adds gating)
3. SwiGLU revisit (different context now with Muon + Peri-LN)
4. HEAD_DIM=64 revisit (different context)
5. DDL residual perturbation (if implementation is clean)
