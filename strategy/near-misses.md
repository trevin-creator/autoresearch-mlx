# Near-Misses

Experiments that were within ~0.01 bpb of the best at time of testing, or that failed in a context that has since changed significantly. These are candidates for revisiting.

Each entry records: the change, the result, the context it was tested in, and what config changes might flip the outcome.

---

## Remove logit soft-capping
- **Result**: 1.626 vs 1.623 baseline (delta: +0.003)
- **Context**: Tested at batch=2^16, warmdown=0.5, MATRIX_LR=0.04
- **Why revisit**: Delta was within noise. Config has changed dramatically since (batch=2^14, warmdown=0.3). The README notes "no logit cap" was a repeated winner on M4 Max. Should be re-tested in current config.
- **Priority**: High

## Cosine warmdown + warmup (0.02)
- **Result**: 1.409 vs 1.402 (delta: +0.007)
- **Context**: Tested at current config (batch=2^14, warmdown=0.3)
- **Why revisit**: Within noise range. The warmup might help or hurt independently of cosine. Could ablate: try cosine-only (no warmup) or warmup-only (no cosine).
- **Priority**: Low

## FINAL_LR_FRAC=0.05
- **Result**: 1.410 vs 1.402 (delta: +0.008)
- **Context**: Tested at current config (batch=2^14, warmdown=0.3)
- **Why revisit**: Close to current best. Might combine well with a different warmdown ratio (e.g., 0.2) or cosine warmdown shape. Low priority as a standalone re-test.
- **Priority**: Low

## SwiGLU activation
- **Result**: 1.665 vs 1.623 baseline (delta: +0.042)
- **Context**: Tested at batch=2^16 with default config. Used 8/3x hidden dim.
- **Why revisit**: Not a near-miss by raw numbers, but was tested in a very different config. The step-count regime (batch=2^14) is completely different. SwiGLU might interact well with width increases. Consider only as part of a bundle.
- **Priority**: Medium (only as combo)

## HEAD_DIM=64
- **Result**: 1.695 vs 1.623 baseline (delta: +0.072)
- **Context**: Tested at batch=2^16
- **Why revisit**: Throughput penalty was the main issue. At batch=2^14, per-step compute matters less since we're already getting 1600+ steps. Worth testing if we increase model width.
- **Priority**: Low (only if increasing width)
