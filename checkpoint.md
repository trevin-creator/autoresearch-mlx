# Checkpoint Review — Experiments 1-13

**Date**: 2026-03-19
**Branch**: autoresearch/mar19
**Best val_bpb**: 1.658510 (experiment 5, batch size 2^13)
**Baseline val_bpb**: 2.126246
**Total improvement**: -0.467736 (22.0% better)

## Metrics

| Metric | Value |
|--------|-------|
| Total experiments | 13 |
| Accepted | 4 (30.8%) |
| Discarded | 9 (69.2%) |
| Crashed | 0 |
| Avg diff lines (all) | 3.0 |
| Avg diff lines (accepted) | 3.0 |
| train.py LOC (baseline) | 526 |
| train.py LOC (current) | 526 |
| LOC growth | 0 |

## Accept rate analysis

4/13 = 30.8%. All 4 accepts were batch size reductions (2^16 → 2^15 → 2^14 → 2^13). Every other dimension explored (depth, LR, warmup, warmdown, activation, head dim) failed. This is a narrow improvement axis.

## Diminishing returns

| Experiment | val_bpb | Delta from previous best |
|-----------|---------|--------------------------|
| Baseline | 2.126 | — |
| Batch 2^15 | 1.908 | -0.218 |
| Batch 2^14 | 1.748 | -0.160 |
| Batch 2^13 | 1.659 | -0.089 |
| Batch 2^12 | 1.693 | +0.035 (regression) |

Clear diminishing returns on batch size reduction. 2^13 is the sweet spot — going smaller (2^12) hurts. The improvement curve is flattening.

## train.py complexity

LOC is unchanged (526). All changes so far were hyperparameter-only (1-2 line edits), except the SwiGLU experiment (11 lines, discarded). No code complexity has been accumulated.

## Strategy assessment

**What worked**: Reducing batch size to get more optimizer steps in the fixed 5-min budget. This small model (11.5M params, depth 4) benefits enormously from more updates.

**What didn't work**:
- Increasing depth (5 or 6): bigger models get too few steps
- LR tuning (matrix LR 0.08, embedding LR 1.2): current LRs are well-tuned
- Schedule tuning (warmup 0.1, warmdown 0.3): current schedule is good
- Architecture changes (SwiGLU, smaller HEAD_DIM): no benefit at this scale

**Stuck signals**: 8 consecutive discards after the batch size wins. Need to try more radical changes.

## Next directions to try

1. **Increase ASPECT_RATIO** (e.g. 96 or 128) to make the model wider while keeping depth 4 — more capacity per step
2. **Remove value embeddings** — simplification that might free up parameters/compute
3. **Remove sliding window attention** — use full causal attention everywhere (simpler, fewer mask computations)
4. **Increase weight decay** (0.2 → 0.4) — stronger regularization with more steps
5. **Try cosine schedule** instead of linear warmdown
6. **Reduce ASPECT_RATIO** to make model smaller/faster for even more steps
