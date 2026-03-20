# Architecture Tweaks for Small Transformers (~50M params)

## 1. MLP Ratio: 4x vs 8/3x (SwiGLU-style) vs Other
- **Paper**: "Llama 2" (Touvron et al., 2023), "OLMo" (Groeneveld et al., 2024)
- **Current setup**: MLP ratio is 4x (`c_fc` is `4 * n_embd`), using SqReLU activation (no gating).
- **Key insight**: The 4x ratio is standard for non-gated MLPs. If switching to a gated activation (SwiGLU/GeGLU), the standard ratio becomes 8/3x (~2.67x) to keep parameter count constant, since gating doubles the up-projection.
- **Actionable options**:
  - Keep 4x with SqReLU (current) -- this is fine
  - Switch to SwiGLU with 8/3x ratio: replace `c_fc` with two projections of size `(8/3)*n_embd`, gate with `silu(x1) * x2`
  - Try a smaller MLP ratio (3x) to free params for more depth

## 2. GQA (Grouped Query Attention)
- **Paper**: "GQA: Training Generalized Multi-Query Attention" (Ainslie et al., 2023)
- **Current setup**: `n_kv_head = n_head` (standard MHA, no GQA). With n_head=2 at depth=4, there's little room for GQA.
- **Key insight**: GQA saves KV parameters, allowing you to either make the model wider or deeper. At this small scale, the savings are minimal. More useful at larger depths.
- **Actionable**: If increasing depth (e.g., DEPTH=8+), set `n_kv_head = n_head // 2` or even `n_kv_head = 1` (MQA) to save memory and params.

## 3. Deeper and Narrower vs Shallower and Wider
- **Papers**: "Scaling Laws" literature, "Depth vs Width" (Nguyen & Salazar, 2019), nanoGPT speedrun community findings (2024)
- **Current**: DEPTH=4, ASPECT_RATIO=64, giving model_dim=256 and ~50M params (much of it in embeddings).
- **Key insight**: At very small scale, deeper models (more layers) tend to learn richer representations but train slower per step. The embedding table (vocab_size * n_embd = 32768 * 256 ~ 8M params) is a large fraction of total params at small widths.
- **Actionable**: Try DEPTH=6 or DEPTH=8 with smaller ASPECT_RATIO to keep total params similar. More layers = more expressiveness. The training will get fewer steps but each step learns more.
  - DEPTH=6, ASPECT_RATIO=43 -> model_dim=256 (rounds to same), 6 layers
  - DEPTH=8, ASPECT_RATIO=32 -> model_dim=256, 8 layers (same width, more depth)
  - DEPTH=12, ASPECT_RATIO=64 -> model_dim=768, ~150M+ params (too big? check memory)

## 4. Tied Embeddings
- **Paper**: "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017), still standard practice
- **Current**: `wte` and `lm_head` are separate. Tying them saves `vocab_size * n_embd` params which can be reallocated to more depth/width.
- **Implementation**: `self.lm_head.weight = self.wte.weight` (share the weight matrix). Need to handle the init carefully.
- **Caution**: The current code has different LR groups for embeddings vs unembeddings, so tying requires choosing one LR. Also, the init is different (wte uses scale 1.0, lm_head uses 0.001).

## 5. Removing Value Embeddings (Simplification)
- **Current**: Value embeddings add a learned per-token bias to attention values on alternating layers. This is from recent "VE" papers.
- **Actionable**: Try removing them entirely. They add `n_ve_layers * vocab_size * kv_dim` parameters. If those params don't help much, removing them and reallocating to more depth could win. This is a simplification experiment.

## 6. HEAD_DIM Tuning
- **Paper**: Various; "Attention Is All You Need" used 64, modern models use 128
- **Current**: HEAD_DIM=128. With model_dim=256, this gives only 2 heads.
- **Actionable**: Try HEAD_DIM=64 to get 4 heads at model_dim=256. More heads = more diverse attention patterns. The compute cost per head is lower but there are more of them.
