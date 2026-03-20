# nanoGPT Speedrun Community Insights (2024-2025)

## Context
The nanoGPT speedrun community (Keller Jordan et al.) systematically optimized small GPT training on CIFAR/FineWeb. Many of the techniques in the current train.py come from this work. Key papers/posts:
- Keller Jordan's "nanoGPT speedrun" series (2024)
- "Modded-NanoGPT" repository
- Karpathy's autoresearch itself inherits from this lineage

## Key Techniques Already in train.py
- QK-norm (present)
- Zero-init on output projections (present: `c_proj` and `mlp.c_proj` init to zero)
- Residual lambdas / x0 skip connections (present)
- Value embeddings on alternating layers (present)
- Logit soft-capping at 15.0 (present)
- Per-parameter-group LR (present: separate LRs for embeddings, matrix, scalar)
- SqReLU activation (present)
- Sliding window attention (present)

## Techniques NOT Yet in train.py (Potential Experiments)

### 1. Muon Optimizer
- **What**: Uses the matrix sign of the momentum (polar decomposition) instead of element-wise Adam update for weight matrices. Dramatically faster convergence.
- **Status**: Comment in train.py says "v0.1: AdamW only. Muon port is future work."
- **Challenge**: Requires SVD/Newton iteration for polar decomposition. MLX has `mx.linalg.svd` but it may be slow. The Newton iteration approach (5 iterations of `X = 1.5*X - 0.5*X@X.T@X`) works on any backend.
- **High potential**: This is the single biggest known win from the speedrun. But implementation complexity is high.

### 2. Embed Shortcircuit (Logit Bypass)
- **What**: Add the token embedding directly to the logits (skip connection from embed to output).
- **Implementation**: `logits = self.lm_head(x) + self.shortcircuit_scale * self.wte(idx) @ self.lm_head.weight.T`
- **Expected impact**: Small but consistent improvement, especially for rare tokens.

### 3. Z-Loss Regularization
- **Paper**: From PaLM (Chowdhery et al., 2022)
- **What**: Add a small penalty on the log of the sum of exponentials of logits: `z_loss = 1e-4 * logsumexp(logits)^2`. Prevents logit drift.
- **Implementation**: After computing logits:
  ```python
  lse = mx.logsumexp(logits, axis=-1)
  z_loss = 1e-4 * mx.mean(lse ** 2)
  loss = ce_loss + z_loss
  ```
- **Already have**: Logit soft-capping at 15.0 serves a similar purpose. May not add much.

### 4. Exponential Moving Average (EMA) of Weights for Eval
- **What**: Maintain an EMA of model weights (decay ~0.999) and use the EMA weights for evaluation.
- **Expected impact**: ~0.5-1% better eval loss for free (only cost is memory for EMA weights).
- **Implementation**: After each optimizer step, update EMA weights.
- **Memory concern**: Doubles parameter memory. At 50M params in bfloat16, that's ~100MB extra. Should be fine on Apple Silicon.

### 5. Token Embedding Scaling
- **What**: Scale token embeddings by `sqrt(n_embd)` after lookup. Standard in original Transformer but often omitted.
- **Current**: No scaling. The init uses scale 1.0 and then norm() is applied.
- **Low priority**: The norm() after embedding already normalizes the scale.
