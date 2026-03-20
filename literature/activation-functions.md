# Activation Function Comparisons

## 1. Squared ReLU (Current)
- **Paper**: "Primer: Searching for Efficient Transformers" (So et al., 2021)
- **Current implementation**: `x = mx.maximum(x, 0) ** 2` (SqReLU)
- **Pros**: Strong sparsity (many activations are exactly zero), good gradient flow for active neurons. Shown to perform well in recent efficient transformers.
- **Cons**: Very aggressive sparsity can hurt at small scale where the model needs all its capacity. The squaring amplifies large activations, which can cause instability.

## 2. SwiGLU
- **Paper**: "GLU Variants Improve Transformer" (Shazeer, 2020); standard in Llama, Mistral, etc. (2023-2024)
- **Key idea**: Replace MLP with gated variant: `output = (silu(W1 @ x) * (W3 @ x)) @ W2`. Requires two up-projections but consistently outperforms ReLU/GELU.
- **Implementation**:
  ```python
  class MLP(nn.Module):
      def __init__(self, config):
          hidden = int(8/3 * config.n_embd)  # or round to multiple of 64
          self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
          self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
          self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
      def __call__(self, x):
          return self.w2(nn.silu(self.w1(x)) * self.w3(x))
  ```
- **Param count**: With 8/3x ratio, roughly equivalent param count to 4x SqReLU MLP. With the same 4x ratio, it would use ~33% more MLP params.
- **Expected impact**: Often 1-3% loss improvement at same compute. The most impactful single architecture change.

## 3. GELU
- **Paper**: "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016). Used in GPT-2, BERT, etc.
- **Implementation**: `nn.gelu(x)` or `nn.gelu_approx(x)` (tanh approximation)
- **Comparison**: Generally slightly worse than SwiGLU but better than plain ReLU. Not worth switching to from SqReLU unless SwiGLU is too complex.

## 4. SiLU (Swish)
- **Paper**: "Searching for Activation Functions" (Ramachandran et al., 2017)
- **Implementation**: `nn.silu(x)` (= x * sigmoid(x))
- **Note**: SiLU alone (without gating) is not commonly used in MLPs. It's primarily used as the gating function in SwiGLU.

## 5. GeGLU (GELU-Gated Linear Unit)
- **Paper**: Same Shazeer (2020) paper as SwiGLU
- **Implementation**: Same as SwiGLU but replace `silu` with `gelu`: `gelu(W1 @ x) * (W3 @ x)`
- **Comparison**: Very similar to SwiGLU. Some papers find it slightly better, others slightly worse. Not worth agonizing over -- either gated variant beats non-gated.

## Recommendation Priority:
1. **Try SwiGLU with 8/3x ratio** -- most likely to improve val_bpb
2. If SwiGLU helps, it's the keeper. If not, SqReLU is fine.
3. Plain GELU or SiLU (non-gated) are unlikely to beat SqReLU.
