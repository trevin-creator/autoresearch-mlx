# Targeted Research Notes — March 19, 2026

Focused literature search on five topics relevant to our ~50M param GPT trained for 5 minutes on MLX.

---

## 1. Weight EMA for Evaluation

### Key Papers
- "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits" (TMLR 2024, arXiv:2411.18704)
- "When, Where and Why to Average Weights?" (arXiv:2502.06761, Feb 2025)

### Findings

**Optimal decay for short training runs:**
- Tested decays: alpha in {0.968, 0.984, 0.992, 0.996, 0.998}.
- For final-model improvement (our use case: EMA only for eval), slower decay (0.996-0.998) is better.
- An averaging window of ~1% of total training budget is optimal. With ~3000 steps in 5 min, that is ~30 steps, implying decay ~0.97. However, since we only use EMA at final eval (not online), higher decay (0.995-0.999) is safe.

**EMA warmup formula:**
```python
ema_decay = min(alpha, (t + 1) / (t + 10))
```
This prevents the EMA from being dominated by early (bad) weights.

**Early training benefit:**
- EMA provides the largest improvement early in training and when learning rate is still high. The gap shrinks as LR decays. Since our warmdown is only 30% of training, there is significant time at high LR where EMA helps.

**Practical recommendation for our setup:**
- Decay = 0.995 to 0.999. Start with 0.998.
- Use the warmup formula above.
- Memory cost: ~100MB extra for 50M params in bfloat16 (or float32 for EMA = 200MB). Fine on Apple Silicon.
- Only swap EMA weights in for the final `evaluate_bpb` call.

**Implementation:**
```python
# After optimizer.update():
ema_decay = min(0.998, (step + 1) / (step + 10))
for path, param in tree_flatten(model.parameters()):
    ema_params[path] = ema_decay * ema_params[path] + (1 - ema_decay) * param.astype(mx.float32)

# Before final eval:
for path, ema_val in ema_params.items():
    set_path_value(model, path, ema_val.astype(mx.bfloat16))
```

**Expected impact:** 0.5-1.5% better val_bpb for essentially free compute.

---

## 2. Tied Embeddings

### Key Papers
- "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017) — the original
- "Super Tiny Language Models" (Guertler, arXiv:2405.14159, 2024)
- OLMo 1B and Gemma both use weight tying

### Findings

**At small scales, tying helps:**
- For our 50M param model with vocab=32768 and n_embd=256, the embedding table is 32768*256 = 8.4M params. The lm_head is another 8.4M. Together they are ~33% of model params.
- Tying saves 8.4M params, which can be reallocated to more depth or width.
- Hugging Face ablation on 1.2B model: tied embeddings performed comparably despite 18% fewer params. When compared to an untied model with the same param count (fewer layers), untied showed *higher* loss.
- GPT-2, OLMo 1B, Gemma all use tying.

**Complications for our codebase:**
- `wte` uses init scale 1.0, `lm_head` uses 0.001. With tying, need to choose one init (use 1.0, since the final `norm(x)` before lm_head controls the scale).
- `wte` has embedding_lr (0.6 * dmodel_scale), `lm_head` has unembedding_lr (0.004 * dmodel_scale). With tying, these become the same parameter. The embedding LR is 150x larger than the unembedding LR. Using the embedding LR for the shared weight may cause instability in the output logits; using the unembedding LR may starve the input embeddings.
- **Mitigation option:** Use an intermediate LR, e.g., geometric mean: sqrt(0.6 * 0.004) ~ 0.05.
- Alternatively, keep them untied but use the saved param budget differently.

**Partial tying / projection-based tying:**
- No recent papers found with clear wins over simple tying for small models.
- The modded-nanogpt speedrun explicitly *unties* head from embedding (separate params).

**Recommendation:**
- Worth testing, but the LR conflict is a real concern. Try tying with a shared LR of ~0.05, or test with the embedding LR and a lower logit softcap to control output scale.
- Expected impact: neutral-to-positive on val_bpb (saves params, slight regularization effect).

---

## 3. Architectural Innovations for Short Training Budgets

### 3a. Z-Loss Regularization
- **Source:** PaLM (Chowdhery et al., 2022)
- **What:** `z_loss = alpha * mean(logsumexp(logits, dim=-1)^2)`
- **Our codebase already has:** logit softcapping at 15.0 via `15.0 * tanh(logits / 15.0)`, which serves a similar purpose (bounds logits).
- **Verdict:** Likely redundant with softcapping. The modded-nanogpt speedrun does NOT use z-loss. Low priority.

### 3b. Embed Shortcircuit (Skip from Embedding to Logits)
- **Source:** modded-nanogpt speedrun (2024-2025)
- **What:** The x0-lambda mechanism already in our train.py IS the embed shortcircuit. Each layer mixes `resid_lambda * x + x0_lambda * x0` where x0 is the initial embedding. This provides a direct gradient path from loss to embeddings through every layer.
- **modded-nanogpt analysis:** x0_lambda reaches ~50% weight at the final layer, meaning the model performs "embedding arithmetic" — adding corrections to the original embedding to produce logits.
- **Our implementation:** x0_lambdas init to 0.1, resid_lambdas init to 1.0. These are learnable.
- **Verdict:** Already implemented. Could experiment with different init values.

### 3c. U-Net Skip Connections (Layer-to-Layer Skips)
- **Source:** modded-nanogpt speedrun
- **What:** Long-range skip connections between symmetric layer pairs (e.g., layer 2->11, 4->10, 6->9 in a 16-layer model). Uses learned scalar gates.
- **Implementation:** `x_decoder = x_decoder + skip_lambda * x_encoder`
- **Our model:** Only 4 layers (DEPTH=4), so U-net skips don't apply. Would become relevant at DEPTH>=8.
- **Verdict:** Not applicable at current depth. Revisit if depth increases.

### 3d. Attention Temperature Scaling
- **Source:** modded-nanogpt speedrun
- **What:** Initial attention scale set to 0.1 (much lower than standard 1/sqrt(d_k)), then dynamically adjusted with window size changes.
- **Our model:** Uses standard `scale = 1.0 / sqrt(head_dim)` = 1/sqrt(128) = 0.088. With QK-norm, the effective scale is already controlled.
- **Verdict:** Our QK-norm + standard scaling is functionally similar. Low priority.

### 3e. Register Tokens / Attention Sinks
- **Source:** "Vision Transformers Need Registers" (Darcet et al., 2024), ICLR 2025 attention sink paper
- **What:** Attention sinks cause the model to dump attention onto the first token. Register tokens provide explicit "no-op" targets.
- **At our scale:** With only 4 layers and 2048 seq len, attention sinks are less of an issue. Register tokens add complexity for minimal gain.
- **Verdict:** Not worth it at current scale.

---

## 4. nanoGPT Speedrun Techniques (Architecture-Specific)

### Already in our train.py:
| Technique | Status |
|-----------|--------|
| QK-norm | Present |
| Zero-init output projections | Present (c_proj, mlp.c_proj) |
| Residual lambdas + x0 skip | Present |
| Value embeddings (alternating) | Present |
| Logit softcapping (tanh) | Present at cap=15 |
| SqReLU activation | Present |
| Sliding window attention | Present |
| Per-param-group LR | Present |

### NOT in our train.py (from the speedrun):

**1. Logit softcap value = 30 (vs our 15):**
- modded-nanogpt uses cap=30 (latest version) or a sigmoid-based variant: `23 * sigmoid((logits + 5) / 7.5)`.
- Our cap of 15.0 is more aggressive. Could try 30.0 to allow more logit dynamic range.
- Low-effort experiment: change `15.0` to `30.0` in two places.

**2. Muon optimizer:**
- Biggest win, but high complexity. Already noted as future work.

**3. FlexAttention with window size warmup:**
- Start with small windows, grow during training. Not easily portable to MLX.

**4. Shifted sigmoid softcap (alternative to tanh):**
- `23 * sigmoid((logits + 5) / 7.5)` — asymmetric, biased positive. Novel but unproven outside the speedrun context.

---

## 5. Normalization Innovations

### 5a. Peri-LN (ICML 2025)
- **Paper:** "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture" (Kim et al., 2025)
- **What:** Apply normalization both before AND after each sub-layer, plus normalize input embeddings and final output.
- **Formula:**
  ```
  x = x + Norm(Attn(Norm(x)))   # norm on both sides
  x = x + Norm(MLP(Norm(x)))
  ```
- **Results:** More stable training, balanced variance growth, consistent improvement across model sizes up to 3.2B.
- **Adopted by:** Gemma 2, OLMo 2.
- **Our model:** Uses pre-norm only: `x = x + attn(norm(x))`. Adding post-norm on the sub-layer output is a simple change.
- **Implementation:**
  ```python
  # In Block.__call__:
  x = x + norm(self.attn(norm(x), ve, mask))  # add outer norm
  x = x + norm(self.mlp(norm(x)))             # add outer norm
  ```
- **Expected impact:** Better training stability, potentially allows higher LR. Low risk.

### 5b. Learnable RMSNorm
- **OLMo finding:** OLMo-0424 used nonparametric layer norm; OLMo 2 switched back to parametric RMSNorm. Ablations showed "no difference" for their scale, but parametric is standard.
- **Our model:** Uses bare RMSNorm (no learnable gamma).
- **Change:** Replace `norm(x)` with `nn.RMSNorm(n_embd)(x)`. Adds n_embd params per norm instance (~10 instances * 256 = 2560 params total, negligible).
- **Evidence for small models:** Embedding layer normalization shows "slight improvement in pre-training loss, more pronounced in smaller models."
- **Verdict:** Low-risk, easy to implement alongside peri-LN.

### 5c. Dynamic Tanh (DyT) — CVPR 2025
- **Paper:** "Transformers without Normalization" (Zhu, Chen, He, LeCun, Liu — FAIR/MIT/Princeton)
- **What:** Replace all normalization layers with `DyT(x) = gamma * tanh(alpha * x) + beta`, where alpha, gamma, beta are learnable per-channel.
- **Results:** Matches or beats RMSNorm across LLaMA, ViT, MAE, DINO, DiT, wav2vec.
- **Advantage:** No mean/variance computation. Computationally cheaper. Fewer ops to compile in MLX.
- **Implementation:**
  ```python
  class DyT(nn.Module):
      def __init__(self, dim):
          super().__init__()
          self.alpha = mx.array(0.5)  # scalar, learnable
          self.gamma = mx.ones((dim,))
          self.beta = mx.zeros((dim,))
      def __call__(self, x):
          return self.gamma * mx.tanh(self.alpha * x) + self.beta
  ```
- **Concern:** Our model already uses norm() inside attention (QK-norm). DyT replaces the normalization purpose but tanh saturation differs from RMSNorm scaling. Would need careful testing.
- **Verdict:** Interesting but higher risk than peri-LN or learnable RMSNorm. Could test as a later experiment.

---

## Summary: Actionable Experiments (Ranked)

| Priority | Experiment | Effort | Expected Impact |
|----------|-----------|--------|-----------------|
| 1 | Weight EMA (decay=0.998) for eval | Low | 0.5-1.5% better val_bpb |
| 2 | Peri-LN (add post-sub-layer norm) | Low | Better stability, possibly lower loss |
| 3 | Learnable RMSNorm (nn.RMSNorm) | Low | Small improvement, more so at small scale |
| 4 | Logit softcap 30 (vs 15) | Trivial | Unknown — may help dynamic range |
| 5 | Tied embeddings (with LR=0.05) | Medium | Saves 8.4M params, neutral-to-positive |
| 6 | DyT normalization | Medium | Potentially faster compile, matched perf |
