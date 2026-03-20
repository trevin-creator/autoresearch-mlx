# Training Efficiency Techniques

## 1. Batch Size Tuning
- **Papers**: "Don't Decay the Learning Rate, Increase the Batch Size" (Smith et al., 2018), "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
- **Current**: TOTAL_BATCH_SIZE = 2**16 = 65536 tokens, DEVICE_BATCH_SIZE = 16 (16 * 2048 = 32768 tokens per micro-batch), so grad_accum_steps = 2.
- **Key insight**: With a 5-minute budget, you want to maximize useful gradient updates. Smaller batch sizes give more steps but noisier gradients. Larger batch sizes give fewer but better steps.
- **Actionable options**:
  - Try TOTAL_BATCH_SIZE = 2**15 (32768) -- eliminates grad accumulation, 2x more steps. Good if the model is undertrained (not enough steps).
  - Try TOTAL_BATCH_SIZE = 2**17 (131072) -- fewer steps but cleaner gradients. Good if the model is already getting enough steps.
  - Try increasing DEVICE_BATCH_SIZE to 32 if memory allows -- reduces overhead from multiple forward passes per step.

## 2. Gradient Accumulation Efficiency
- **Current**: Accumulates gradients across 2 micro-batches, then divides by grad_accum_steps.
- **Key insight**: Each micro-batch requires a separate forward+backward pass. With only 2 accum steps, the overhead is minimal. But if grad_accum_steps=1 (no accumulation), you save the tree_map addition and division operations.
- **Actionable**: If reducing TOTAL_BATCH_SIZE to 2**15, set DEVICE_BATCH_SIZE=16 to get grad_accum_steps=1.

## 3. Larger Device Batch Size
- **Current**: DEVICE_BATCH_SIZE=16. MLX on Apple Silicon has good memory bandwidth but launch overhead per operation.
- **Actionable**: Try DEVICE_BATCH_SIZE=32 or 64. This increases the tokens processed per forward pass, potentially improving GPU utilization. Check that memory doesn't blow up.

## 4. Reducing the Overhead
- **Current**: FINAL_EVAL_BATCH_SIZE=256. This is only for the final assessment, so it doesn't affect training time budget.
- **Actionable**: Not a priority since assessment time is excluded from the training budget.

## 5. Compilation / Graph Optimization
- **MLX-specific**: MLX lazily processes operations and can fuse them. Calling mx.eval() less frequently (batching more computation before processing) can help.
- **Current**: The code calls mx.eval(loss, grads) after each micro-batch, then mx.eval(model.parameters(), *optimizer.state) after each step. This is already reasonable.
- **Caution**: Don't delay mx.eval too long or memory will grow from unrealized computation graphs.
