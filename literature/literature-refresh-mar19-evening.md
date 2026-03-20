# Literature Refresh — March 19 Evening

Three parallel research agents covering: novel activations, Muon advances, and short-budget training.

---

## HIGH PRIORITY FINDINGS

### 1. mx.compile for the training step (MLX-specific, huge potential)

Our train.py does NOT use `mx.compile`. Compiling the forward-backward-update into a single fused graph eliminates kernel launch overhead for all the element-wise ops (norms, activations, gates, tanh capping). MLX docs report significant speedups. With grad_accum_steps=1 (our case), this is straightforward.

Implementation requires restructuring the loop to be pure (no side effects inside compiled fn). The optimizer state management also needs to be compatible with compilation.

**Expected impact**: More steps in 5 minutes = direct val_bpb improvement. Could be 5-15% more steps.

### 2. mx.fast.rms_norm instead of inline norm()

Our inline `norm(x) = x * rsqrt(mean(x*x) + eps)` compiles to multiple kernels. `mx.fast.rms_norm(x, weight=None, eps=1e-5)` uses a single optimized Metal kernel. Called ~20+ times per forward pass. Drop-in replacement.

### 3. xIELU activation (arXiv:2411.13010)

Exact formula: `xIELU(x) = x * sigmoid(beta * x) + alpha * min(0, x)^2`
- beta: trainable per-channel, init 1.0 (controls sigmoid sharpness)
- alpha: trainable per-channel, init 0.0 (controls negative-side leaky quadratic)
- When alpha=0, beta=1: reduces to SiLU
- 2 extra params per hidden dim (negligible at ~1536 total params)
- Beat SqReLU by ~1-3% at small scale in the paper
- Drop-in replacement for the activation function

### 4. Warm-start Newton-Schulz (Muon optimization)

Cache previous step's orthogonalized matrix as starting point for next step. Allows reducing from 5 to 2-3 iterations in steady state since consecutive momentum matrices are similar. Could recover ~30-40% of Muon's compute overhead.

### 5. Nesterov momentum in Muon

Replace standard momentum with Nesterov before orthogonalization. Small but consistent ~0.5-1% convergence improvement.

---

## MEDIUM PRIORITY FINDINGS

### 6. GLU + SqReLU (gated SqReLU)

`sqrelu(W1(x)) * W3(x)` with hidden_dim reduced to 8/3x for param parity. Gets gating benefit while keeping our working activation. Cerebras-GPT found SqReLU-GLU roughly equivalent to SwiGLU at small scale.

### 7. Sparse Attention Gate (nanoGPT speedrun Record 28)

Sigmoid gate on first 12 dims of residual stream modulates each attention head's output. Reduces attention sink behavior. Simple, adds ~12 * n_head params.

### 8. SiLU^2 (Squared SiLU)

`silu(x)^2` — gets SqReLU-like sharpness with SiLU's smooth gradients. Zero extra params, drop-in replacement.

### 9. Cautious Muon (sign-agreement masking WITHOUT rescaling)

Apply sign-agreement mask on top of Muon's orthogonalized update without the harmful rescaling factor. Different from our earlier Cautious Adam test (which used rescaling and was terrible).

---

## LOWER PRIORITY

- Multi-token prediction auxiliary head (nanoGPT Record 53)
- Deep Delta Learning (arXiv:2601.00417) — only ~0.7% at 124M, our resid_lambdas similar
- NorMuon — weight decay scheduling refinement
- Bigram hash embeddings (nanoGPT Record 62)

## NOT APPLICABLE

- Curriculum learning, data importance sampling: require modifying prepare.py
- Progressive layer growth: compile overhead too expensive
- Decoupled embeddings: too expensive at 15.7M scale
- SOAP optimizer: worse than Muon at our scale
