# Learning Rate Schedule Improvements

## 1. WSD (Warmup-Stable-Decay) Schedule
- **Paper**: "Scaling Data-Constrained Language Models" (Muennighoff et al., NeurIPS 2023); popularized by MiniCPM (2024) and Llama 3 training
- **Key idea**: Instead of cosine decay, use three phases: short warmup, long stable LR plateau, then sharp linear/cosine decay at the end. The stable phase lets the model explore longer at peak LR.
- **Current setup**: Already uses a WSD-like schedule (warmup -> flat -> cooldown via `WARMUP_RATIO=0.0`, `WARMDOWN_RATIO=0.5`). The warmdown is 50% of training.
- **Actionable tweak**: Try reducing `WARMDOWN_RATIO` to 0.2-0.3 to spend more time at peak LR. With only 5 minutes, more time at full LR may help. Also try `WARMDOWN_RATIO=0.667` (longer decay) to see if smoother convergence helps.

## 2. Trapezoidal / Warmup-Stable-Decay with Sqrt Decay
- **Paper**: "WSD-S: Taking the Warmup Out" (2024 blog posts from Eleuther/community)
- **Key idea**: Replace linear warmdown with `1/sqrt(t)` decay in the cooldown phase, which decays fast initially then slows — gets you lower LR sooner but doesn't go to zero too fast.
- **Implementation**: In `get_lr_multiplier`, change the cooldown formula from linear to sqrt:
  ```python
  cooldown_progress = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
  return (1.0 - cooldown_progress) ** 0.5 * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC
  ```

## 3. Cosine Cooldown (instead of linear)
- **Paper**: Standard from Chinchilla (Hoffmann et al., 2022), still competitive in 2024 ablations
- **Key idea**: Replace the linear cooldown with cosine: `0.5 * (1 + cos(pi * cooldown_progress))`. Smoother transition often helps.
- **Implementation**: In `get_lr_multiplier`:
  ```python
  cooldown_progress = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
  return 0.5 * (1 + math.cos(math.pi * cooldown_progress)) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC
  ```

## 4. Short Warmup is Fine for Small Models
- **Paper**: Various 2024 scaling studies confirm warmup can be very short (<1% of training)
- **Current**: `WARMUP_RATIO=0.0` (no warmup). This works with the QK-norm + zero-init projection setup since gradients are well-behaved at init.
- **Actionable**: Try `WARMUP_RATIO=0.02` (tiny warmup). With zero init on projections it probably doesn't matter, but worth a quick test.

## 5. FINAL_LR_FRAC > 0
- **Paper**: Llama 3 training report; "Minicpm: Unveiling the potential of small language models" (2024)
- **Key idea**: Don't decay LR all the way to 0. Setting `FINAL_LR_FRAC=0.05-0.1` keeps the model learning at the end of training.
- **Current**: `FINAL_LR_FRAC=0.0`. Try 0.05 or 0.1.
