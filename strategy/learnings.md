# Learnings

Hardware- and config-specific insights derived from experiments on this machine.
Updated after every experiment. Each entry explains *why* something worked or didn't, not just whether it did.

---

## Muon optimizer is a breakthrough (high confidence)

Muon (matrix sign of momentum via Newton-Schulz iteration) for weight matrices gave 1.344 vs 1.363 — a 0.020 improvement despite losing ~100 steps (1492 vs 1588) to the iteration overhead. This proves that per-step quality CAN overcome step-count loss when the optimizer improvement is large enough.

Muon uses beta1=0.95 (higher momentum than Adam's 0.8) and 5 Newton-Schulz iterations per step. It replaces AdamW for 2D weight matrices only; embeddings, scalars, and gates still use Adam.

**Implication**: The "per-step compute overhead kills everything" rule has an exception — when the change dramatically improves optimization quality per step. Muon is the biggest known optimizer win from the nanoGPT speedrun and it carries over to MLX.

## Per-step compute overhead is the dominant constraint (high confidence, but with exception)

At this model scale (11.5M params, dim=256), the model is so fast per step that ANY added computation — even tiny things like z-loss logsumexp, EMA weight updates, or an extra multiply for embed shortcircuit — costs measurable steps. Results:
- Z-loss: 1.406 (lost ~70 steps, delta +0.004)
- EMA: 1.452 (lost ~40 steps, but also averaged in stale weights)
- Embed shortcircuit: 1.415 (lost ~50 steps)
- Removing logit cap: 1.430 (GAINED 194 steps but worse per-step learning)

The last point is key: logit cap removal added steps but hurt quality, proving that per-step quality and step count both matter. The current model sits at a precise balance point. Winning changes must either: (1) improve per-step quality WITHOUT adding compute, or (2) add so much per-step quality that the step loss is worth it (which means a big architectural shift, not a small regularizer).

**Implication**: Focus on changes that are compute-neutral (e.g., different weight initialization, different optimizer schedule shape) or on bigger architectural shifts where the quality-per-step gain is large enough to overcome step loss.

## Single-machine, sequential-only constraint (hard constraint)

All experiments run on one M4 Max laptop. No parallel training jobs — concurrent runs produce incomparable results due to GPU contention and non-deterministic scheduling. This means every experiment costs ~7 minutes of wall time with no shortcut. The strategy knowledge base exists specifically to compensate: invest thinking time (consulting hypotheses, interactions, near-misses) to make each of those 7-minute slots count. Bad experiment picks are the main bottleneck, not compute.

---

## Step-count dominance (high confidence)

This model is **step-count-limited**, not gradient-quality-limited. On this hardware (M4 Max), reducing batch size from 2^16 → 2^15 → 2^14 gave the two largest improvements in the entire run (1.623 → 1.540 → 1.402). The model is small enough (~11.5M params) that each forward/backward pass is fast, so more optimizer steps per 5-minute budget wins.

**However**, there's a floor: 2^13 (device_batch=4) degraded to 1.542 — gradient noise dominates at that scale. The sweet spot appears to be 2^14 with device_batch=8.

**Implication**: Any change that increases per-step compute (larger model, more complex optimizer) must be justified by proportionally better per-step learning. Changes that reduce per-step compute (simpler architecture, fewer params) could be viable if they allow even more steps without hitting the noise floor.

## MATRIX_LR=0.04 is near-optimal at batch=2^14 (high confidence)

Three data points: LR=0.02→1.412, LR=0.04→1.402, LR=0.06→1.495. The optimum is at or very near 0.04. The sqrt-scaling hypothesis (predict ~0.028) was wrong — this model apparently benefits from relatively high LR even at smaller batch. The sensitivity is asymmetric: too high hurts much more than too low. 0.03 might yield a marginal gain but the expected value is small.

**Implication**: Don't waste runs on further MATRIX_LR tuning. Focus on other dimensions. If architecture changes significantly (depth, width, activation), LR may need re-tuning, but 0.04 is a good default.

## Full attention beats sliding window at 4 layers (high confidence)

Changing WINDOW_PATTERN from "SSSL" to "LLLL" gave 1.394 vs 1.402 — an improvement with MORE steps (1645 vs 1600). Two wins: (1) full context visibility on all layers is more expressive, (2) uniform mask = better caching = slightly faster per step. At 4 layers, restricting 3 of them to half-context windows is too aggressive.

**Implication**: Simplifications that remove unnecessary constraints can win on both quality and speed. The sliding window pattern makes more sense at higher depth where some layers can afford limited context.

## WARMDOWN_RATIO=0.3 is the sweet spot (high confidence)

Three data points: 0.5→1.623, 0.3→1.613 (at old batch), 0.2→1.429 (at batch=2^14). The 0.5→0.3 change was a clear win. But 0.3→0.2 was a big regression at the current config. The model needs sufficient cooldown time — ~30% of training spent in LR decay appears optimal. Not worth tuning further.

## Peri-LN is the biggest single-experiment architectural win (high confidence)

Adding post-sub-layer normalization (`x = x + norm(attn(norm(x)))`) gave 1.363 vs 1.387 — a 0.024 improvement with negligible compute cost (~1588 vs 1611 steps). This stabilizes variance growth through the residual stream. Adopted by Gemma 2 and OLMo 2.

**Implication**: The model benefits from tighter variance control. Other normalization experiments (learnable RMSNorm: worse, DyT: much worse) suggest that the specific combination of fixed RMSNorm + peri-LN placement is important — not just any normalization will do.

## Logit softcap = 15 is better than 30 with peri-LN (medium confidence)

Softcap 30 gave 1.370 vs softcap 15 at 1.363. The tighter cap works better, possibly because peri-LN already controls variance so the model doesn't need wider logit range.

## Logit soft-capping helps training quality (high confidence)

Removing logit capping gives more steps (1794 vs 1600 — the tanh costs compute) but worse val_bpb (1.430 vs 1.402). The cap constrains logit magnitude which stabilizes gradients and improves per-step learning efficiency. This is NOT dead weight — it's an active training quality improvement. Do not remove.

**Implication**: The cap value (15.0) could potentially be tuned but removing it entirely is harmful.

## SwiGLU needs param-count compensation (medium confidence)

SwiGLU with 8/3x hidden dim reduced params from ~11.5M to ~11.6M (actually similar) but val_bpb was 1.665 vs 1.623. The issue may not be param count but rather that the model dim (256) is too small for gating to help — the gate projection consumes capacity that could be used for raw width. At larger model dims, SwiGLU is standard. Consider testing only if we increase model width.

## Gradient centralization hurts here (medium confidence)

GC gave 1.692 vs 1.613. Centering gradients may conflict with the existing QK-norm + RMSNorm setup, or the model may be too small for the implicit regularization to help.

## Value embeddings: more is better (high confidence)

VE on all 4 layers (instead of alternating 2) gave 1.388 vs 1.394 — a clear win. Added 4.2M params (11.5M → 15.7M) with negligible compute cost (embedding lookups are fast on MLX). VE is one of the most efficient ways to add model capacity because it's just a lookup table, not a matrix multiply.

**Implication**: VE is high-value-per-FLOP capacity. If we need more model capacity, VE expansion is the cheapest way to get it. However, removing VE entirely was devastating (1.543 at 7.3M params), confirming it's load-bearing.

## OLD: Value embeddings are critical at this scale (high confidence)

Removing VE dropped params from 11.5M to 7.3M (VE = 36% of total params!) and val_bpb from 1.402 to 1.543. VE is not dead weight — it provides substantial model capacity via input-dependent value bias. At dim=256 with only 4 layers, these extra embedding parameters are essential.

**Implication**: Don't try to simplify VE away. If anything, VE could be expanded (e.g., add VE to every layer instead of alternating). But the current alternating pattern is a reasonable efficiency trade-off.

## EMA of weights hurts at this training length (medium confidence)

EMA with decay=0.99 (1.452) and 0.995 (1.496) both worse than raw weights (1.402). Two reasons: (1) EMA update costs compute per step, reducing total steps from ~1600 to ~1560. (2) The linear warmdown schedule already produces a well-converged final checkpoint — the last weights ARE the best weights. EMA averages in earlier, worse checkpoints.

**Implication**: Don't use EMA unless the training schedule changes (e.g., constant LR with no warmdown, where the final checkpoint is noisy). The current warmdown serves the same purpose as EMA — it produces smooth convergence.

## Cautious Adam is too aggressive with rescaling (medium confidence)

C-Adam with the standard rescaling factor gave 1.846 vs 1.613 — a dramatic regression. The mask * (size/sum(mask)) rescaling amplifies surviving updates too much. A version *without* the rescaling factor might be worth trying, but the signal is strongly negative.

## HEAD_DIM=64 hurts throughput more than it helps expressiveness (low confidence)

4 heads gave 1.695 vs 1.623 with fewer steps (351 vs 400). The smaller head dim increased compute per step enough to reduce step count, and the extra attention heads didn't compensate. Tested at baseline config though — might differ at current batch=2^14.

## FINAL_LR_FRAC=0.05 is slightly worse (low confidence)

Gave 1.410 vs 1.402 — close but consistently worse. The model benefits from full LR decay to zero. Not worth revisiting unless the warmdown shape changes.
