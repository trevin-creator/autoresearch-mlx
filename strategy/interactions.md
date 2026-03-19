# Parameter Interactions

Known and suspected couplings between hyperparameters and architecture choices.
Consult this before testing any change in isolation — if the parameter is coupled, consider adjusting its partner simultaneously.

---

## Confirmed Interactions

### Batch size ↔ Learning rate
- **Evidence**: At batch=2^14: LR=0.02→1.412, LR=0.04→1.402, LR=0.06→1.495. The original LR=0.04 (tuned for batch=2^16) is still optimal at batch=2^14. Sqrt-scaling rule did NOT hold — the model wants high LR regardless of batch size.
- **Rule of thumb**: Sqrt scaling is a starting guess, not gospel. For this model, LR=0.04 appears robust across batch sizes 2^14 to 2^16. The coupling is weaker than expected.
- **Action**: After architecture changes, still re-tune LR. But for batch-only changes, 0.04 is likely fine.

### Batch size ↔ Device batch size
- **Constraint**: TOTAL_BATCH_SIZE must be divisible by (DEVICE_BATCH_SIZE * MAX_SEQ_LEN). When reducing total batch, must also reduce device batch if it would violate this.
- **Action**: Mechanical constraint — just ensure the assertion passes.

### WARMDOWN_RATIO ↔ FINAL_LR_FRAC
- **Suspected**: These control the same thing (end-of-training LR behavior). Changing one may shift the optimal value of the other.
- **Evidence**: FINAL_LR_FRAC=0.05 was slightly worse at warmdown=0.3. Might be different at warmdown=0.2.

---

## Suspected Interactions

### Model width ↔ Activation function
- **Hypothesis**: SwiGLU gating may only help above a certain model width. At dim=256, the gate projection may waste capacity. At dim=384+, gating provides useful multiplicative interactions.
- **Evidence**: SwiGLU failed at dim=256. Standard practice uses SwiGLU at dim ≥ 768.

### Model depth ↔ Step count
- **Hypothesis**: More layers = more compute per step = fewer steps in budget. The crossover point where depth's expressiveness beats step count is unknown for this hardware.
- **Evidence**: depth=3 (wider) gave 1.784 vs depth=4 at 1.623 — shallower was worse despite more steps (334 vs 400). But depth=3 also used a very different model_dim (384 vs 256).

### Weight decay ↔ Batch size
- **Hypothesis**: Smaller batches provide implicit regularization through gradient noise. Explicit weight decay may need to decrease as batch size decreases.
- **Evidence**: None yet. Worth testing.

### HEAD_DIM ↔ Per-step compute
- **Hypothesis**: Smaller head dim increases attention compute. At step-count-dominated regimes, this trade-off is unfavorable.
- **Evidence**: HEAD_DIM=64 reduced steps from 400 → 351 at old config. Effect at current config unknown.
