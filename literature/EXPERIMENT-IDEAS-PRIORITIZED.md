# Prioritized Experiment Ideas

Sorted by expected impact / implementation effort ratio. All are compatible with the current 5-minute MLX training setup.

---

## Tier 1: High Impact, Low Effort (try first)

### 1. SwiGLU Activation (replace SqReLU)
- **Source**: Shazeer (2020), standard in Llama/Mistral/Gemma
- **What**: Replace the MLP with a gated variant using SiLU gating
- **Change**: In MLP, add a gate projection (w3), use `silu(w1(x)) * w3(x)`, reduce hidden dim to `int(8/3 * n_embd)` (round to multiple of 64) to keep param count similar
- **Why it works**: Gating provides multiplicative interactions that SqReLU lacks. Consistently 1-3% better across all scales.

### 2. Cautious Adam (C-Adam)
- **Source**: arXiv:2411.16085 (Liang et al., 2024)
- **What**: Mask optimizer updates where gradient and momentum disagree in sign
- **Change**: Add 2 lines in `_step`: compute `mask = (update * grad_f32 > 0)`, apply `update = update * mask`
- **Why it works**: Prevents the optimizer from taking steps in directions where the gradient has recently reversed. Zero overhead.

### 3. HEAD_DIM=64 (more attention heads)
- **Source**: Empirical; more heads = more diverse attention patterns
- **What**: Change HEAD_DIM from 128 to 64. At model_dim=256, this gives 4 heads instead of 2.
- **Change**: One constant: `HEAD_DIM = 64`
- **Why it works**: 2 heads is very few. 4 heads allows more diverse attention patterns with negligible compute cost.

### 4. Reduce WARMDOWN_RATIO to 0.3
- **Source**: WSD schedule literature, MiniCPM (2024)
- **What**: Spend more training time at peak LR
- **Change**: `WARMDOWN_RATIO = 0.3` (from 0.5)
- **Why it works**: With a short 5-min budget, spending 50% of training in warmdown may be too conservative. More time at peak LR = more learning.

### 5. Gradient Centralization
- **Source**: Yong et al. (ECCV 2020), widely used in 2024
- **What**: Subtract the mean of each gradient row for 2D+ params
- **Change**: One line in `_step`: `if grad_f32.ndim >= 2: grad_f32 = grad_f32 - mx.mean(grad_f32, axis=-1, keepdims=True)`
- **Why it works**: Implicit regularization, projects gradients onto a constraint surface. Zero overhead.

---

## Tier 2: Medium Impact, Medium Effort

### 6. Increase Depth (DEPTH=6 or 8, adjust ASPECT_RATIO)
- **Source**: Depth vs Width scaling literature
- **What**: More layers at same or smaller width
- **Change**: Try DEPTH=8, ASPECT_RATIO=32 (keeps model_dim=256, adds more layers). Or DEPTH=6, ASPECT_RATIO=43 (similar).
- **Trade-off**: Fewer steps in 5 minutes (more compute per step), but each step is more expressive. Worth testing.

### 7. GQA at Larger Depths
- **Source**: Ainslie et al. (2023)
- **What**: If increasing depth, use n_kv_head < n_head to save params for more layers
- **Change**: Set `n_kv_head = max(1, n_head // 2)` in GPTConfig
- **Synergy**: Combine with experiment 6.

### 8. Cosine Cooldown (instead of linear)
- **Source**: Standard Chinchilla-style
- **What**: Replace linear warmdown with cosine curve
- **Change**: In `get_lr_multiplier`, use `0.5 * (1 + cos(pi * cooldown_progress))` instead of linear interpolation
- **Why it works**: Smoother LR transition, spends more time near peak before dropping.

### 9. Learnable RMSNorm
- **Source**: Zhang & Sennrich (2019)
- **What**: Add learnable scale parameters to normalization
- **Change**: Replace `norm(x)` with `nn.RMSNorm(n_embd)` instances in Block
- **Why it works**: Gives the model per-channel control over feature magnitudes. Tiny param increase.

### 10. FINAL_LR_FRAC = 0.05
- **Source**: Llama 3 training, MiniCPM
- **What**: Don't decay LR to zero
- **Change**: `FINAL_LR_FRAC = 0.05`
- **Why it works**: Maintains some learning capacity at end of training. Especially useful for short runs.

---

## Tier 3: High Impact but High Complexity

### 11. Muon Optimizer (for weight matrices)
- **Source**: nanoGPT speedrun (Keller Jordan, 2024)
- **What**: Use matrix sign of momentum via Newton iteration for weight matrices, keep Adam for embeddings
- **Change**: Major optimizer rewrite. Add Newton iteration: `X = 1.5*X - 0.5*X@X.T@X` (5 iterations)
- **Why it works**: Biggest single win in the speedrun (~10-15% loss improvement). But requires careful implementation and may be slow on MLX due to matrix-matrix multiplies in the optimizer.
- **Risk**: MLX SVD/matmul overhead in optimizer may eat into the training time budget.

### 12. EMA of Weights for Final Assessment
- **Source**: Standard practice, Polyak averaging
- **What**: Maintain EMA (decay=0.995) of model weights, use for assessment
- **Change**: After each optimizer step, update EMA. Before assessment, swap in EMA weights.
- **Cost**: ~100MB extra memory for 50M params in bfloat16.

### 13. Schedule-Free Adam
- **Source**: arXiv:2405.15682 (Defazio & Mishchenko, ICML 2024)
- **What**: Eliminate LR schedule, use iterate averaging
- **Change**: Major optimizer rewrite with dual parameter tracking
- **Risk**: Complex, memory-hungry, may not help with 5-min budget.

---

## Tier 4: Quick Tests (low effort, uncertain impact)

### 14. Remove embedding norm
- Try removing `x = norm(x)` after `self.wte(idx)` -- simplification experiment.

### 15. Remove value embeddings
- Remove VE entirely and see if it hurts. If not, the freed params can go to more depth.

### 16. Smaller total batch size (2**15)
- Eliminates grad accumulation, doubles step count. Good if model needs more updates.

### 17. beta1 tuning
- Try ADAM_BETAS = (0.85, 0.95) or (0.7, 0.95) for more/less momentum.

### 18. Weight decay tuning
- Try WEIGHT_DECAY = 0.1 (less regularization) or 0.3 (more).
