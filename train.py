"""
Autoresearch pretraining script. Single-device, single-file.
Apple Silicon MLX port of karpathy/autoresearch.
Usage: uv run train.py
"""

import gc
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_str(name, default):
    return os.environ.get(name, default)


def env_betas(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    beta1, beta2 = raw.split(",")
    return (float(beta1), float(beta2))


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
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
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


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


def get_machine_id():
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        chip = result.stdout.strip()
        if chip:
            return chip
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = result.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass
    return "unknown"


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
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
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask.astype(q.dtype))
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2
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

    def init_weights(self):
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5

        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(
            mx.bfloat16
        )

        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(
                -scale, scale, block.attn.c_q.weight.shape
            ).astype(mx.bfloat16)
            block.attn.c_k.weight = mx.random.uniform(
                -scale, scale, block.attn.c_k.weight.shape
            ).astype(mx.bfloat16)
            block.attn.c_v.weight = mx.random.uniform(
                -scale, scale, block.attn.c_v.weight.shape
            ).astype(mx.bfloat16)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight).astype(mx.bfloat16)
            block.mlp.c_fc.weight = mx.random.uniform(
                -scale, scale, block.mlp.c_fc.weight.shape
            ).astype(mx.bfloat16)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight).astype(mx.bfloat16)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight).astype(
                    mx.bfloat16
                )

        self.resid_lambdas = mx.ones((self.config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1, dtype=mx.float32)

        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-scale, scale, ve.weight.shape).astype(mx.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(char in "SL" for char in pattern)
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
            x = self.resid_lambdas[i].astype(x.dtype) * x + self.x0_lambdas[i].astype(x.dtype) * x0
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


polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003104),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum,
    lr,
    weight_decay,
    beta2,
    ns_steps,
):
    rows, cols = stacked_params.shape[-2:]
    grads = stacked_grads.astype(momentum_buffer.dtype)
    momentum_buffer = momentum * momentum_buffer + (1 - momentum) * grads
    g = (1 - momentum) * grads + momentum * momentum_buffer

    x = g.astype(mx.bfloat16)
    x_norm = mx.linalg.norm(x.astype(mx.float32), axis=(-2, -1), keepdims=True)
    x = x / (x_norm.astype(x.dtype) * 1.02 + 1e-6)
    if rows > cols:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            xt = mx.swapaxes(x, -2, -1)
            a_mat = xt @ x
            b_mat = b * a_mat + c * (a_mat @ a_mat)
            x = a * x + x @ b_mat
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            xt = mx.swapaxes(x, -2, -1)
            a_mat = x @ xt
            b_mat = b * a_mat + c * (a_mat @ a_mat)
            x = a * x + b_mat @ x

    g = x.astype(mx.float32)
    red_dim = -1 if rows >= cols else -2
    red_dim_size = cols if red_dim == -1 else rows
    v_mean = mx.mean(g * g, axis=red_dim, keepdims=True)
    v_norm = mx.sqrt(mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size)

    second_momentum_f32 = second_momentum_buffer.astype(mx.float32)
    second_momentum_f32 = beta2 * second_momentum_f32 + (1 - beta2) * v_mean
    step_size = mx.rsqrt(mx.maximum(second_momentum_f32, 1e-10))
    scaled_sq_sum = (v_mean * red_dim_size) * (step_size * step_size)
    v_norm_new = mx.sqrt(mx.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
    g = g * final_scale

    lr_scaled = lr * max(1.0, rows / cols) ** 0.5
    params_f32 = stacked_params.astype(mx.float32)
    mask = (g * params_f32) >= 0
    new_params = (
        params_f32
        - lr_scaled * g
        - lr_scaled * weight_decay * params_f32 * mask.astype(params_f32.dtype)
    )
    return (
        new_params.astype(stacked_params.dtype),
        momentum_buffer,
        second_momentum_f32.astype(second_momentum_buffer.dtype),
    )


class MuonAdamW:
    def __init__(
        self,
        model,
        unembedding_lr,
        embedding_lr,
        matrix_lr,
        weight_decay,
        adam_betas,
        scalar_lr,
        use_muon=True,
    ):
        self.adam_config = {}
        self.adam_state = {}
        self.muon_groups = []
        self.muon_state = {}

        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        muon_groups_by_shape = {}
        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            if "blocks" in path and param.ndim == 2:
                if use_muon:
                    muon_groups_by_shape.setdefault(param.shape, []).append(path)
                else:
                    # Controlled first pass: keep matrix params on the same raw
                    # LR and weight decay as the Muon path so only the optimizer changes.
                    self.adam_config[path] = {
                        "lr": matrix_lr,
                        "betas": adam_betas,
                        "eps": 1e-10,
                        "weight_decay": weight_decay,
                    }
            elif "wte" in path:
                self.adam_config[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "value_embeds" in path:
                self.adam_config[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "lm_head" in path:
                self.adam_config[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "resid_lambdas" in path:
                self.adam_config[path] = {
                    "lr": scalar_lr * 0.01,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "x0_lambdas" in path:
                self.adam_config[path] = {
                    "lr": scalar_lr,
                    "betas": (0.96, 0.95),
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            else:
                self.adam_config[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }

        for shape in sorted(muon_groups_by_shape):
            paths = muon_groups_by_shape[shape]
            self.muon_groups.append(
                {
                    "paths": paths,
                    "lr": matrix_lr,
                    "initial_lr": matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": MUON_NS_STEPS,
                    "beta2": MUON_BETA2,
                    "weight_decay": weight_decay,
                }
            )

        self.adam_initial_lrs = {path: config["lr"] for path, config in self.adam_config.items()}

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
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

    def _adamw_step(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        weight_decay = config["weight_decay"]

        if path not in self.adam_state:
            self.adam_state[path] = {
                "m": mx.zeros_like(grad_f32),
                "v": mx.zeros_like(grad_f32),
                "t": 0,
            }

        state = self.adam_state[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * weight_decay)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def _muon_step(self, model, group, flat_grads, flat_params):
        paths = group["paths"]
        if not all(path in flat_grads for path in paths):
            return

        stacked_grads = mx.stack([flat_grads[path] for path in paths])
        stacked_params = mx.stack([flat_params[path] for path in paths])
        state_key = tuple(paths)
        if state_key not in self.muon_state:
            num_params, rows, cols = stacked_params.shape
            state_shape = (num_params, rows, 1) if rows >= cols else (num_params, 1, cols)
            self.muon_state[state_key] = {
                "momentum_buffer": mx.zeros_like(stacked_params),
                "second_momentum_buffer": mx.zeros(state_shape, dtype=stacked_params.dtype),
            }

        state = self.muon_state[state_key]
        new_params, state["momentum_buffer"], state["second_momentum_buffer"] = muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            momentum=group["momentum"],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            beta2=group["beta2"],
            ns_steps=group["ns_steps"],
        )
        for idx, path in enumerate(paths):
            self._set_path_value(model, path, new_params[idx])

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.adam_config:
                continue
            config = self.adam_config[path]
            param = flat_params[path]
            new_param = self._adamw_step(path, grad, param, config)
            self._set_path_value(model, path, new_param)
        for group in self.muon_groups:
            self._muon_step(model, group, flat_grads, flat_params)

    def set_lr_multiplier(self, multiplier):
        for path, config in self.adam_config.items():
            config["lr"] = self.adam_initial_lrs[path] * multiplier
        for group in self.muon_groups:
            group["lr"] = group["initial_lr"] * multiplier

    def set_muon_momentum(self, momentum):
        for group in self.muon_groups:
            group["momentum"] = momentum

    def set_muon_weight_decay(self, weight_decay):
        for group in self.muon_groups:
            group["weight_decay"] = weight_decay

    @property
    def state(self):
        arrays = []
        for state in self.adam_state.values():
            arrays.extend([state["m"], state["v"]])
        for group in self.muon_groups:
            state = self.muon_state.get(tuple(group["paths"]))
            if state is not None:
                arrays.extend([state["momentum_buffer"], state["second_momentum_buffer"]])
        return arrays


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = env_int("AUTORESEARCH_ASPECT_RATIO", 64)
HEAD_DIM = env_int("AUTORESEARCH_HEAD_DIM", 128)
WINDOW_PATTERN = env_str("AUTORESEARCH_WINDOW_PATTERN", "L")

TOTAL_BATCH_SIZE = env_int("AUTORESEARCH_TOTAL_BATCH_SIZE", 2**16)
EMBEDDING_LR = env_float("AUTORESEARCH_EMBEDDING_LR", 0.6)
UNEMBEDDING_LR = env_float("AUTORESEARCH_UNEMBEDDING_LR", 0.004)
MATRIX_LR = env_float("AUTORESEARCH_MATRIX_LR", 0.04)
MUON_NS_STEPS = env_int("AUTORESEARCH_MUON_NS_STEPS", 5)
MUON_BETA2 = env_float("AUTORESEARCH_MUON_BETA2", 0.95)
SCALAR_LR = env_float("AUTORESEARCH_SCALAR_LR", 0.5)
WEIGHT_DECAY = env_float("AUTORESEARCH_WEIGHT_DECAY", 0.2)
ADAM_BETAS = env_betas("AUTORESEARCH_ADAM_BETAS", (0.8, 0.95))
WARMUP_RATIO = env_float("AUTORESEARCH_WARMUP_RATIO", 0.0)
WARMDOWN_RATIO = env_float("AUTORESEARCH_WARMDOWN_RATIO", 0.5)
FINAL_LR_FRAC = env_float("AUTORESEARCH_FINAL_LR_FRAC", 0.0)
MOMENTUM_SCHEDULE = env_str("AUTORESEARCH_MOMENTUM_SCHEDULE", "baseline")
WEIGHT_DECAY_SCHEDULE = env_str("AUTORESEARCH_WEIGHT_DECAY_SCHEDULE", "linear")

# Model size
DEPTH = env_int("AUTORESEARCH_DEPTH", 4)
DEVICE_BATCH_SIZE = env_int("AUTORESEARCH_DEVICE_BATCH_SIZE", 16)
FINAL_EVAL_BATCH_SIZE = env_int("AUTORESEARCH_FINAL_EVAL_BATCH_SIZE", 256)
STARTUP_EXCLUDE_STEPS = env_int("AUTORESEARCH_STARTUP_EXCLUDE_STEPS", 1)
SEED = env_int("AUTORESEARCH_SEED", 42)
USE_MUON = env_str("AUTORESEARCH_OPTIMIZER", "muon").lower() == "muon"


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    if MOMENTUM_SCHEDULE == "constant_095":
        return 0.95
    if MOMENTUM_SCHEDULE == "flat":
        frac = min(step / 600, 1)
        return (1 - frac) * 0.90 + frac * 0.95
    if MOMENTUM_SCHEDULE == "fast_ramp":
        frac = min(step / 150, 1)
        return (1 - frac) * 0.85 + frac * 0.95
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    if WEIGHT_DECAY_SCHEDULE == "constant":
        return WEIGHT_DECAY
    if WEIGHT_DECAY_SCHEDULE == "slow_linear":
        return WEIGHT_DECAY * (1 - 0.5 * progress)
    return WEIGHT_DECAY * (1 - progress)


t_start = time.time()
mx.random.seed(SEED)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
t_data = time.time()
print(f"Data/tokenizer loaded in {t_data - t_start:.1f}s")

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
model.init_weights()
mx.eval(model.parameters())
# Materialize the fixed training masks before tracing so compile never mutates the cache.
model._get_masks(MAX_SEQ_LEN)
mx.eval(*tuple(model._mask_cache.values()))
num_params = sum(param.size for _, param in tree_flatten(model.parameters()))

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = MuonAdamW(
    model,
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
    scalar_lr=SCALAR_LR,
    use_muon=USE_MUON,
)

_loss_grad_fn = nn.value_and_grad(model, lambda inputs, targets: model(inputs, targets=targets))
compiled_state = [model.state]


@partial(mx.compile, inputs=compiled_state, outputs=compiled_state)
def loss_grad_fn(inputs, targets):
    return _loss_grad_fn(inputs, targets)


print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

smooth_train_loss = 0.0
total_training_time = 0.0
step_times = []
step_toksec = []
step = 0
t_compiled = None

while True:
    t0 = time.time()
    accum_grads = None
    train_loss = None

    for _ in range(grad_accum_steps):
        loss, grads = loss_grad_fn(x, y)
        mx.eval(loss, grads)
        if t_compiled is None:
            t_compiled = time.time()
            print(f"Startup finished in {t_compiled - t_data:.1f}s")
        train_loss = loss
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda lhs, rhs: lhs + rhs, accum_grads, grads)
        x, y, epoch = next(train_loader)

    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda grad: grad * (1.0 / grad_accum_steps), accum_grads)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    optimizer.set_lr_multiplier(lrm)
    optimizer.set_muon_momentum(get_muon_momentum(step))
    optimizer.set_muon_weight_decay(get_weight_decay(progress))
    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), *optimizer.state)

    train_loss_f = float(train_loss.item())
    if train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    dt = time.time() - t0
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt
        step_times.append(dt)
        step_toksec.append(tok_per_sec)

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    remaining = max(0.0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step >= STARTUP_EXCLUDE_STEPS and total_training_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training completed in {t_train - t_compiled:.1f}s")

timed_steps = max(step - STARTUP_EXCLUDE_STEPS, 0)
total_tokens = timed_steps * TOTAL_BATCH_SIZE
print("Starting final eval...")
print(f"Final eval batch size: {FINAL_EVAL_BATCH_SIZE}")
val_bpb = evaluate_bpb(model, tokenizer, FINAL_EVAL_BATCH_SIZE)
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

steady_state_mfu = 0.0
peak_vram_mb = get_peak_memory_mb()
data_load_seconds = round(t_data - t_start, 1)
compile_seconds = round(t_compiled - t_data, 1) if t_compiled is not None else 0.0
avg_toksec = round(statistics.mean(step_toksec)) if step_toksec else 0
median_toksec = round(statistics.median(step_toksec)) if step_toksec else 0
avg_step_ms = round(statistics.mean(step_times) * 1000, 1) if step_times else 0.0
median_step_ms = round(statistics.median(step_times) * 1000, 1) if step_times else 0.0
optimizer_name = "muon" if USE_MUON else "adamw"
machine_id = get_machine_id()
record = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "commit": get_git_commit(),
    "seed": SEED,
    "optimizer": optimizer_name,
    "machine": machine_id,
    "mlx_version": mx.__version__,
    "depth": DEPTH,
    "aspect_ratio": ASPECT_RATIO,
    "head_dim": HEAD_DIM,
    "window_pattern": WINDOW_PATTERN,
    "num_params_M": round(num_params / 1e6, 1),
    "matrix_lr": MATRIX_LR,
    "effective_matrix_lr": MATRIX_LR,
    "weight_decay": WEIGHT_DECAY,
    "weight_decay_schedule": WEIGHT_DECAY_SCHEDULE,
    "adam_betas": list(ADAM_BETAS),
    "momentum_schedule": MOMENTUM_SCHEDULE,
    "total_batch_size": TOTAL_BATCH_SIZE,
    "dmodel_lr_scale": round((model_dim / 768) ** -0.5, 6),
    "matrix_lr_uses_dmodel_scale": False,
    "val_bpb": round(val_bpb, 6),
    "num_steps": step,
    "timed_steps": timed_steps,
    "total_tokens_M": round(total_tokens / 1e6, 1),
    "data_load_seconds": data_load_seconds,
    "compile_seconds": compile_seconds,
    "training_seconds": round(total_training_time, 1),
    "total_seconds": round(t_eval - t_start, 1),
    "peak_vram_mb": round(peak_vram_mb, 1),
    "avg_tok_sec": avg_toksec,
    "median_tok_sec": median_toksec,
    "avg_step_ms": avg_step_ms,
    "median_step_ms": median_step_ms,
}
jsonl_path = os.environ.get("AUTORESEARCH_BENCH_LOG", "/tmp/autoresearch_bench.jsonl")

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"timed_steps:      {timed_steps}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"seed:             {SEED}")
print(f"optimizer:        {optimizer_name}")
print(f"machine:          {machine_id}")
print(f"data_load_sec:    {data_load_seconds}")
print(f"compile_sec:      {compile_seconds}")
print(f"avg_tok_sec:      {avg_toksec}")
print(f"median_tok_sec:   {median_toksec}")
print(f"avg_step_ms:      {avg_step_ms}")
print(f"median_step_ms:   {median_step_ms}")

with open(jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")

print(f"Result appended to {jsonl_path}")
print("JSON:" + json.dumps(record))
