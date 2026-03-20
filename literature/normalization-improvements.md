# Normalization Improvements

## 1. QK-Norm (Already Present)
- **Paper**: "Scaling Vision Transformers to 22B Parameters" (Dehghani et al., 2023); adopted widely in 2024
- **Current setup**: The code applies `norm()` (RMSNorm without learnable scale) to both Q and K after RoPE: `q = norm(self.rope(q))`, `k = norm(self.rope(k))`.
- **Key insight**: QK-norm prevents attention logit growth and enables higher LRs. This is already implemented correctly.
- **No change needed** -- this is a good baseline.

## 2. Pre-Norm vs Post-Norm
- **Current**: Pre-norm (norm before attention/MLP in each block): `x + attn(norm(x))`.
- **Papers**: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020), various 2024 papers
- **Key insight**: Pre-norm is more stable for training but can lead to "representation collapse" in deep networks where later layers contribute less. Post-norm is harder to train but can be better for deep models.
- **Not actionable**: At DEPTH=4, pre-norm is fine. Post-norm would require careful LR tuning.

## 3. Learnable RMSNorm Scale
- **Current**: The `norm()` function is a bare RMSNorm without a learnable gain parameter: `x * rsqrt(mean(x^2) + eps)`.
- **Paper**: Standard RMSNorm (Zhang & Sennrich, 2019) includes a learnable per-channel scale `gamma`.
- **Actionable**: Add a learnable `gamma` parameter to the norm. Use `nn.RMSNorm(n_embd)` instead of the custom `norm()` function. This adds negligible parameters but gives the model more flexibility.
- **Implementation**: Replace `norm(x)` calls in Block with `self.norm1(x)` and `self.norm2(x)` where these are `nn.RMSNorm(config.n_embd)`.
- **Caution**: The current code also applies `norm` to embeddings and final output. Those may benefit from learnable scale too.

## 4. Deep Norm (for deeper models)
- **Paper**: "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., 2022)
- **Key idea**: Scale residual connections by `alpha` and init certain weights smaller. Helps train very deep transformers.
- **Current**: Already has `resid_lambdas` and `x0_lambdas` which serve a similar purpose.
- **Not immediately actionable** unless going to DEPTH >= 16.

## 5. Norm Before Logits
- **Current**: Applies `norm(x)` before `lm_head(x)` (the final projection).
- **This is correct** and standard practice. No change needed.

## 6. Removing the Initial Embedding Norm
- **Current**: `x = self.wte(idx)` followed by `x = norm(x)`.
- **Consideration**: Some papers (2024 OLMo, Llama) do NOT norm after embedding lookup. The wte init already controls the scale. Removing this norm is a simplification worth testing.
