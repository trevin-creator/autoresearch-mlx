# Optimizer Improvements Compatible with AdamW

## 1. Cautious Adam / C-Adam (arXiv:2411.16085)
- **Paper**: "Cautious Optimizers: Improving Training with One Line of Code" (Liang et al., 2024)
- **Key idea**: Mask out optimizer update components where the gradient and momentum disagree in sign. Only apply the update where `sign(grad) == sign(momentum)`. This prevents the optimizer from "overshooting" when the gradient direction has changed.
- **Implementation**: In `_step`, after computing the Adam update, add:
  ```python
  update = state["m"] / denom
  mask = (update * grad_f32 > 0).astype(mx.float32)
  update = update * mask
  param_f32 = param_f32 - step_size * update
  ```
- **Expected impact**: 0.5-2% loss improvement with zero cost. Pure upside.

## 2. GrokFast: EMA Gradient Filtering (arXiv:2405.20233)
- **Paper**: "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" (Lee et al., 2024)
- **Key idea**: Maintain an EMA of gradients and amplify the slow-changing component. Encourages learning generalizable patterns over memorization.
- **Implementation**: Before the Adam update, filter gradients:
  ```python
  # In optimizer, maintain gradient EMA
  if "grad_ema" not in state:
      state["grad_ema"] = mx.zeros_like(grad_f32)
  state["grad_ema"] = 0.98 * state["grad_ema"] + 0.02 * grad_f32
  grad_f32 = grad_f32 + 5.0 * state["grad_ema"]  # amplify slow component
  ```
- **Caution**: The amplification factor (5.0) and EMA decay (0.98) need tuning. Start conservative (factor=2.0, decay=0.95).

## 3. Schedule-Free AdamW (arXiv:2405.15682)
- **Paper**: "The Road Less Scheduled" (Defazio & Mishchenko, ICML 2024 Oral)
- **Key idea**: Eliminate the LR schedule entirely. Maintain two sequences: the iterate `z` (used for gradient computation) and the averaged iterate `x` (used for evaluation). Uses a specific interpolation formula.
- **Implementation complexity**: Moderate — requires maintaining extra parameter copies and modifying the training loop. The formula is:
  ```python
  # z_t = param - lr * adam_update(grad)
  # x_t = (1 - 1/(t+1)) * x_{t-1} + 1/(t+1) * z_t
  # Use z for training forward passes, x for evaluation
  ```
- **Risk**: Higher memory (2x params). May not be worth it at 50M scale with 5-min budget. Complex to implement correctly.

## 4. Gradient Centralization
- **Paper**: "Gradient Centralization: A New Optimization Technique for DNNs" (Yong et al., ECCV 2020), still widely used in 2024
- **Key idea**: Subtract the mean of each gradient tensor before the optimizer step. For weight matrices: `grad = grad - grad.mean(axis=-1, keepdims=True)`. Acts as implicit regularization.
- **Implementation**: One line per gradient in `_step`:
  ```python
  if grad_f32.ndim >= 2:
      grad_f32 = grad_f32 - mx.mean(grad_f32, axis=-1, keepdims=True)
  ```
- **Expected impact**: Small but consistent improvement, zero cost.

## 5. Adam with Decoupled Weight Decay Scaling
- **Key idea**: The current code applies `weight_decay` scaled by `lr`. Some implementations decouple this so weight decay doesn't change with the LR schedule. Currently the code does `param * (1 - lr * weight_decay)` which couples them.
- **Actionable**: Try decoupling: `param * (1 - base_lr * weight_decay)` using the initial LR rather than the scheduled LR. This keeps regularization constant even during warmdown.

## 6. Higher beta1 for Short Runs
- **Paper**: Empirical finding from nanoGPT speedrun community (2024)
- **Key idea**: With short training runs, momentum (beta1) can be lower to be more responsive to recent gradients. Current `ADAM_BETAS=(0.8, 0.95)` already uses a lower beta1 than default (0.9). Could try even lower: (0.7, 0.95) or (0.85, 0.95).
