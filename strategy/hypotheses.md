# Hypotheses

Untested experiment ideas with theoretical rationale. Includes both single changes and multi-change bundles.
Each entry tracks: the hypothesis, expected mechanism, dependencies, and status.

---

## Active Hypotheses

### H1: MATRIX_LR tuning — try 0.03
- **Change**: MATRIX_LR from 0.04 → 0.03
- **Rationale**: We now have three data points: 0.02→1.412, 0.04→1.402, 0.06→1.495. The optimum is near 0.04 but 0.03 hasn't been tested and sits between the two closest points.
- **Risk**: Low — but expected gain is small since 0.04 already looks near-optimal
- **Status**: Untested (0.02 and 0.06 tested and discarded)
- **Priority**: LOW (deprioritized — diminishing returns likely)

### H2: Reduce weight decay
- **Change**: WEIGHT_DECAY from 0.2 → 0.1
- **Rationale**: With more steps and smaller batches, the model sees more gradient updates. Higher weight decay may be over-regularizing. The README notes "lower regularization" was a winner on M4 Max #2.
- **Risk**: Low
- **Status**: Untested
- **Priority**: HIGH

### H3: Cosine warmdown (instead of linear)
- **Change**: Replace linear warmdown with `0.5 * (1 + cos(pi * cooldown_progress))`
- **Rationale**: Cosine spends more effective time near peak LR before decaying. Combined with the already-shortened warmdown window (0.3), this could extract more learning from the peak phase.
- **Risk**: Low — well-understood technique
- **Status**: Untested
- **Priority**: MEDIUM

### H4: Remove logit soft-capping (re-test in current config)
- **Change**: Remove `logits = 15.0 * mx.tanh(logits / 15.0)`
- **Rationale**: Near-miss at old config (+0.003). Current config is very different (4x more steps). README shows this was a repeated winner.
- **Status**: Untested (revisit from near-misses)
- **Priority**: MEDIUM

### H5: Add small warmup (WARMUP_RATIO=0.02)
- **Change**: WARMUP_RATIO from 0.0 → 0.02
- **Rationale**: With 1600 steps, 0.02 warmup = ~32 steps. The first step is excluded already (STARTUP_EXCLUDE_STEPS=1), but a brief ramp may stabilize early training and prevent the optimizer from overshooting on initial noisy gradients.
- **Risk**: Very low
- **Status**: Untested
- **Priority**: MEDIUM

### H6: Increase depth with compensating width reduction
- **Change**: DEPTH=6, ASPECT_RATIO=43 (→ model_dim=256, same as now, but 6 layers instead of 4)
- **Rationale**: More depth = more expressive per-step. If per-step time increase is modest, the expressiveness gain may outweigh the step-count loss. Key question: does 6-layer at ~1100 steps beat 4-layer at ~1600 steps?
- **Risk**: Medium — could lose too many steps
- **Status**: Untested
- **Priority**: MEDIUM

### H7: SwiGLU + wider model bundle
- **Change**: SwiGLU activation + ASPECT_RATIO=96 (→ model_dim=384) + DEPTH=4 + adjusted LR
- **Rationale**: SwiGLU alone failed partly because model_dim=256 may be too narrow for gating to help. At dim=384 with SwiGLU's 8/3x hidden (=1024), the MLP has more capacity. Needs LR re-tuning since architecture is very different.
- **Risk**: High — multiple interacting changes, may need sequential LR sweep (2-3 extra runs at ~7 min each)
- **Dependencies**: Should try H1 (LR tuning) first to establish LR sensitivity
- **Status**: Untested
- **Priority**: LOW (save for later)

### H8: ADAM_BETAS tuning
- **Change**: Try (0.85, 0.95) or (0.9, 0.95)
- **Rationale**: With more steps and smaller batches, less momentum (higher beta1) might help the optimizer respond faster to the noisier gradients. Or more momentum (0.85) might smooth things out. Unclear direction — needs testing.
- **Risk**: Low
- **Status**: Untested
- **Priority**: MEDIUM

### H9: Remove value embeddings
- **Change**: Remove VE entirely (no value_embeds dict, no ve parameter in attention)
- **Rationale**: VE adds an embedding lookup per layer for alternating layers. If it doesn't help at this scale, removing it frees compute for more steps and simplifies the model. Aligns with simplicity criterion.
- **Risk**: Low — easy to test and revert
- **Status**: Untested
- **Priority**: MEDIUM

### H10: Reduce WARMDOWN_RATIO further to 0.2
- **Change**: WARMDOWN_RATIO from 0.3 → 0.2
- **Rationale**: The 0.5 → 0.3 change was a clear win. Pushing further may extract more. At 1600 steps, 0.2 warmdown = ~320 steps of decay, which is still plenty.
- **Risk**: Very low
- **Status**: Untested
- **Priority**: MEDIUM

---

## Tested / Resolved

(Move hypotheses here after testing, with outcome and reference to results.tsv)
