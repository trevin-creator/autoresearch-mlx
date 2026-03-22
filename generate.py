"""
Text generation with the trained GPT model.
First run 'uv run train.py' to train and save weights, then run this.

Usage: uv run generate.py "Your prompt here"
       uv run generate.py                      # interactive mode
"""

import math
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from prepare import MAX_SEQ_LEN, Tokenizer


# ---------------------------------------------------------------------------
# Model definition (copied from train.py to avoid importing the training loop)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve, mask):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = norm(self.rope(q))
        k = norm(self.rope(k))
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.silu(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.zeros((config.n_layer,), dtype=mx.float32)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        }
        self._mask_cache = {}

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, seq_len):
        unique_windows = set(self.window_sizes)
        for window_size in unique_windows:
            key = (seq_len, window_size)
            if key not in self._mask_cache:
                if window_size >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(seq_len, window_size)
        return [self._mask_cache[(seq_len, window_size)] for window_size in self.window_sizes]

    def __call__(self, idx, targets=None, reduction="mean"):
        _, seq_len = idx.shape
        masks = self._get_masks(seq_len)
        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, masks[i])
        x = norm(x)
        logits = self.lm_head(x).astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)
        if targets is None:
            return logits
        valid = targets != -1
        targets_safe = mx.where(valid, targets, mx.zeros_like(targets))
        ce = nn.losses.cross_entropy(logits, targets_safe, reduction="none")
        ce = ce * valid
        if reduction == "none":
            return ce
        denom = mx.maximum(mx.sum(valid), 1)
        return mx.sum(ce) / denom


# ---------------------------------------------------------------------------
# Hyperparameters (must match train.py!)
# ---------------------------------------------------------------------------
ASPECT_RATIO = 64
HEAD_DIM = 128
DEPTH = 4
WINDOW_PATTERN = "SSSL"

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.safetensors")


def load_model(tokenizer):
    """Build model and load trained weights."""
    vocab_size = tokenizer.get_vocab_size()
    model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=model_dim // HEAD_DIM,
        n_kv_head=model_dim // HEAD_DIM,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

    model = GPT(config)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"FEHLER: Keine trainierten Gewichte gefunden unter {WEIGHTS_PATH}")
        print(f"Bitte zuerst trainieren: uv run train.py")
        sys.exit(1)

    print("Lade trainierte Gewichte...")
    weights = mx.load(WEIGHTS_PATH)
    # Manually assign weights since tree_unflatten doesn't handle list indices
    for key, value in weights.items():
        parts = key.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)
    mx.eval(model.parameters())

    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Modell geladen: {num_params / 1e6:.1f}M Parameter (depth={DEPTH}, dim={model_dim})")
    return model


def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=40):
    """Generate text token by token."""
    tokens = list(tokenizer.encode(prompt))

    print(f"\n{'='*50}")
    print(f"Prompt: \"{prompt}\"")
    print(f"Tokens: {max_tokens}, Temperatur: {temperature}, Top-k: {top_k}")
    print(f"{'='*50}\n")
    print(prompt, end="", flush=True)

    for _ in range(max_tokens):
        context = tokens[-MAX_SEQ_LEN:]
        x = mx.array([context])
        logits = model(x)[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                top_k_vals = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                threshold = top_k_vals[:, -1:]
                logits = mx.where(logits < threshold, float("-inf"), logits)
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs).item()
        else:
            next_token = mx.argmax(logits, axis=-1).item()

        tokens.append(next_token)
        print(tokenizer.decode([next_token]), end="", flush=True)

    print("\n")


def main():
    print("Lade Tokenizer...")
    tokenizer = Tokenizer.from_directory()
    model = load_model(tokenizer)

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        generate(model, tokenizer, prompt)
    else:
        print("\n Interaktiver Modus - gib einen Text ein und das Modell")
        print(" vervollstaendigt ihn. (Ctrl+C zum Beenden)\n")
        print(" HINWEIS: Das Modell hat nur 11.5M Parameter und wurde")
        print(" 5 Minuten trainiert. Erwarte keinen ChatGPT-Level!\n")

        while True:
            try:
                prompt = input("Prompt> ")
                if not prompt.strip():
                    continue
                generate(model, tokenizer, prompt)
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
