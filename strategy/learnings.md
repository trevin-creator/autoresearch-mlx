# Learnings

Hardware- and config-specific insights derived from experiments on this machine.
Updated after every experiment. Each entry explains *why* something worked or didn't, not just whether it did.

---

## Single-machine, sequential-only constraint (hard constraint)

All experiments run on one M4 Max laptop. No parallel training jobs — concurrent runs produce incomparable results due to GPU contention and non-deterministic scheduling. This means every experiment costs ~7 minutes of wall time with no shortcut. The strategy knowledge base exists specifically to compensate: invest thinking time (consulting hypotheses, interactions, near-misses) to make each of those 7-minute slots count. Bad experiment picks are the main bottleneck, not compute.

---

## Step-count dominance (high confidence)

This model is **step-count-limited**, not gradient-quality-limited. On this hardware (M4 Max), reducing batch size from 2^16 → 2^15 → 2^14 gave the two largest improvements in the entire run (1.623 → 1.540 → 1.402). The model is small enough (~11.5M params) that each forward/backward pass is fast, so more optimizer steps per 5-minute budget wins.

**However**, there's a floor: 2^13 (device_batch=4) degraded to 1.542 — gradient noise dominates at that scale. The sweet spot appears to be 2^14 with device_batch=8.

**Implication**: Any change that increases per-step compute (larger model, more complex optimizer) must be justified by proportionally better per-step learning. Changes that reduce per-step compute (simpler architecture, fewer params) could be viable if they allow even more steps without hitting the noise floor.

## LR and batch size are tightly coupled (high confidence)

MATRIX_LR=0.06 at batch_size=2^14 was significantly worse (1.495 vs 1.402). The original 0.04 was tuned for 2^16. With 4x fewer tokens per batch, the effective gradient noise is higher, so the LR that worked before may already be at or above optimal. Future LR experiments should try *lower*, not higher.

## Warmdown ratio matters at short budgets (medium confidence)

Reducing WARMDOWN_RATIO from 0.5 to 0.3 gave a small but real win (1.623 → 1.613). Spending 50% of a 5-minute budget in LR decay is too conservative. The current 0.3 (70% at peak LR) seems reasonable. Could try 0.2 but diminishing returns likely.

## Logit soft-capping is neutral at this scale (low confidence)

Removing logit capping gave 1.626 vs 1.623 baseline — within noise. Worth revisiting at the current config (batch=2^14, warmdown=0.3) since the context has changed significantly.

## SwiGLU needs param-count compensation (medium confidence)

SwiGLU with 8/3x hidden dim reduced params from ~11.5M to ~11.6M (actually similar) but val_bpb was 1.665 vs 1.623. The issue may not be param count but rather that the model dim (256) is too small for gating to help — the gate projection consumes capacity that could be used for raw width. At larger model dims, SwiGLU is standard. Consider testing only if we increase model width.

## Gradient centralization hurts here (medium confidence)

GC gave 1.692 vs 1.613. Centering gradients may conflict with the existing QK-norm + RMSNorm setup, or the model may be too small for the implicit regularization to help.

## Cautious Adam is too aggressive with rescaling (medium confidence)

C-Adam with the standard rescaling factor gave 1.846 vs 1.613 — a dramatic regression. The mask * (size/sum(mask)) rescaling amplifies surviving updates too much. A version *without* the rescaling factor might be worth trying, but the signal is strongly negative.

## HEAD_DIM=64 hurts throughput more than it helps expressiveness (low confidence)

4 heads gave 1.695 vs 1.623 with fewer steps (351 vs 400). The smaller head dim increased compute per step enough to reduce step count, and the extra attention heads didn't compensate. Tested at baseline config though — might differ at current batch=2^14.

## FINAL_LR_FRAC=0.05 is slightly worse (low confidence)

Gave 1.410 vs 1.402 — close but consistently worse. The model benefits from full LR decay to zero. Not worth revisiting unless the warmdown shape changes.
