from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPYX_SRC = ROOT / "spyx" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SPYX_SRC) not in sys.path:
    sys.path.insert(0, str(SPYX_SRC))

import haiku as hk
import jax
import jax.numpy as jnp
import mlx.core as mx

import spyx.nn as spyx_nn
import spyx.axn as spyx_axn
from spyx_mlx.nn import ALIF as MlxALIF
from spyx_mlx.nn import CuBaLIF as MlxCuBaLIF
from spyx_mlx.nn import IF as MlxIF
from spyx_mlx.nn import LIF as MlxLIF
from spyx_mlx.nn import RLIF as MlxRLIF


@dataclass
class BenchResult:
    neuron: str
    config: str
    batch: int
    hidden: int
    steps: int
    backend: str
    mean_seconds: float
    std_seconds: float
    steps_per_second: float
    ms_per_step: float


def _state_for(neuron: str, batch: int, hidden: int, xp: str):
    if neuron in {"ALIF", "CuBaLIF"}:
        shape = (batch, hidden * 2)
    else:
        shape = (batch, hidden)
    if xp == "jax":
        return jnp.zeros(shape, dtype=jnp.float32)
    return mx.zeros(shape, dtype=mx.float32)


def _spyx_transformed(neuron: str, hidden: int):
    hidden_shape = (hidden,)
    if neuron == "IF":
        klass = spyx_nn.IF
        kwargs = {"threshold": 1.0}
    elif neuron == "LIF":
        klass = spyx_nn.LIF
        kwargs = {"beta": 0.9, "threshold": 1.0}
    elif neuron == "ALIF":
        klass = spyx_nn.ALIF
        kwargs = {"beta": 0.9, "gamma": 0.9, "threshold": 1.0}
    elif neuron == "CuBaLIF":
        klass = spyx_nn.CuBaLIF
        kwargs = {"alpha": 0.8, "beta": 0.9, "threshold": 1.0}
    elif neuron == "RLIF":
        # Equivalent to spyx.nn.RLIF, but hoists parameters outside scan to
        # avoid tracer leaks in benchmark-only JAX transforms.
        threshold = 1.0

        def model(xs, s0):
            recurrent = hk.get_parameter(
                "w",
                hidden_shape * 2,
                init=hk.initializers.TruncatedNormal(),
            )
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(0.9))
            beta = jnp.clip(beta, 0.0, 1.0)
            spike_fn = spyx_axn.superspike()

            def step_fn(v, x_t):
                spikes = spike_fn(v - threshold)
                feedback = spikes @ recurrent
                v = beta * v + x_t + feedback - spikes * threshold
                return v, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
    else:
        raise ValueError(f"Unsupported neuron: {neuron}")

    def model(xs, s0):
        cell = klass(hidden_shape=hidden_shape, **kwargs)

        def step_fn(state, x_t):
            out, new_state = cell(x_t, state)
            return new_state, out

        sf, ys = jax.lax.scan(step_fn, s0, xs)
        return ys, sf

    return hk.without_apply_rng(hk.transform(model))


def _mlx_runner(neuron: str, hidden: int):
    hidden_shape = (hidden,)
    if neuron == "IF":
        cell = MlxIF(hidden_shape=hidden_shape, threshold=1.0)
    elif neuron == "LIF":
        cell = MlxLIF(hidden_shape=hidden_shape, beta_init=0.9, threshold=1.0)
    elif neuron == "ALIF":
        cell = MlxALIF(hidden_shape=hidden_shape, beta_init=0.9, gamma_init=0.9, threshold=1.0)
    elif neuron == "CuBaLIF":
        cell = MlxCuBaLIF(hidden_shape=hidden_shape, alpha_init=0.8, beta_init=0.9, threshold=1.0)
    elif neuron == "RLIF":
        cell = MlxRLIF(hidden_shape=hidden_shape, beta_init=0.9, threshold=1.0)
    else:
        raise ValueError(f"Unsupported neuron: {neuron}")

    def run(xs, s0):
        outs = []
        s = s0
        for t in range(xs.shape[0]):
            y, s = cell(xs[t], s)
            outs.append(y)
        return mx.stack(outs, axis=0), s

    if hasattr(mx, "compile"):
        run = mx.compile(run)
    return run


def _time_jax(neuron: str, batch: int, hidden: int, steps: int, repeats: int):
    rng = np.random.default_rng(0)
    xs_np = rng.standard_normal((steps, batch, hidden), dtype=np.float32)
    xs = jnp.array(xs_np)
    s0 = _state_for(neuron, batch, hidden, xp="jax")

    transformed = _spyx_transformed(neuron, hidden)
    params = transformed.init(jax.random.PRNGKey(0), xs, s0)
    fn = jax.jit(transformed.apply)
    use_jit = True

    try:
        y, s = fn(params, xs, s0)
        jax.block_until_ready(y)
        jax.block_until_ready(s)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        if jax.default_backend().lower() == "metal" and "default_memory_space" in err:
            use_jit = False
            fn = transformed.apply
            y, s = fn(params, xs, s0)
            jax.block_until_ready(y)
            jax.block_until_ready(s)
        else:
            raise

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y, s = fn(params, xs, s0)
        jax.block_until_ready(y)
        jax.block_until_ready(s)
        times.append(time.perf_counter() - t0)
    return np.array(times, dtype=np.float64)


def _time_mlx(neuron: str, batch: int, hidden: int, steps: int, repeats: int):
    rng = np.random.default_rng(0)
    xs_np = rng.standard_normal((steps, batch, hidden), dtype=np.float32)
    xs = mx.array(xs_np)
    s0 = _state_for(neuron, batch, hidden, xp="mlx")
    fn = _mlx_runner(neuron, hidden)

    y, s = fn(xs, s0)
    mx.eval(y, s)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y, s = fn(xs, s0)
        mx.eval(y, s)
        times.append(time.perf_counter() - t0)
    return np.array(times, dtype=np.float64)


def _summary(neuron: str, cfg_name: str, batch: int, hidden: int, steps: int, backend: str, times: np.ndarray):
    mean_s = float(times.mean())
    std_s = float(times.std())
    return BenchResult(
        neuron=neuron,
        config=cfg_name,
        batch=batch,
        hidden=hidden,
        steps=steps,
        backend=backend,
        mean_seconds=mean_s,
        std_seconds=std_s,
        steps_per_second=float(steps / mean_s),
        ms_per_step=float(mean_s * 1000.0 / steps),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark spyx (JAX) vs spyx_mlx (MLX)")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--json", type=Path, default=Path("bench_spyx_vs_mlx.json"))
    parser.add_argument("--mlx-device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    args = parser.parse_args()

    if args.mlx_device == "cpu":
        mx.set_default_device(mx.cpu)
    elif args.mlx_device == "gpu":
        mx.set_default_device(mx.gpu)

    configs = [
        ("small", 32, 128, 128),
        ("medium", 64, 256, 128),
        ("large", 96, 384, 128),
    ]
    neurons = ["IF", "LIF", "ALIF", "CuBaLIF", "RLIF"]
    jax_backend_name = f"spyx-jax-{jax.default_backend().lower()}"
    mlx_backend_name = "spyx-mlx-gpu" if "gpu" in str(mx.default_device()).lower() else "spyx-mlx-cpu"

    print(f"JAX backend: {jax.default_backend()}")
    print(f"MLX default device: {mx.default_device()}")

    rows: list[BenchResult] = []
    failures: list[dict[str, str]] = []
    for neuron in neurons:
        for cfg_name, batch, hidden, steps in configs:
            try:
                if not args.skip_jax:
                    jax_times = _time_jax(neuron, batch, hidden, steps, args.repeats)
                    rows.append(_summary(neuron, cfg_name, batch, hidden, steps, jax_backend_name, jax_times))
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "backend": jax_backend_name,
                    "neuron": neuron,
                    "config": cfg_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

            try:
                if not args.skip_mlx:
                    mlx_times = _time_mlx(neuron, batch, hidden, steps, args.repeats)
                    rows.append(_summary(neuron, cfg_name, batch, hidden, steps, mlx_backend_name, mlx_times))
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "backend": mlx_backend_name,
                    "neuron": neuron,
                    "config": cfg_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

            print(f"done: {neuron} {cfg_name}")

    payload = {
        "run_config": {
            "repeats": args.repeats,
            "mlx_device": args.mlx_device,
            "skip_jax": args.skip_jax,
            "skip_mlx": args.skip_mlx,
            "jax_backend": jax.default_backend(),
            "mlx_default_device": str(mx.default_device()),
        },
        "results": [asdict(r) for r in rows],
        "failures": failures,
    }
    args.json.write_text(json.dumps(payload, indent=2))

    print("\nSummary (ms/step, lower is better):")
    print(f"neuron\tconfig\t{jax_backend_name}\t{mlx_backend_name}\tspeedup(mlx_vs_jax)")
    for neuron in neurons:
        for cfg_name, batch, hidden, steps in configs:
            j = next((r for r in rows if r.neuron == neuron and r.config == cfg_name and r.backend == jax_backend_name), None)
            m = next((r for r in rows if r.neuron == neuron and r.config == cfg_name and r.backend == mlx_backend_name), None)
            if j is None or m is None:
                print(f"{neuron}\t{cfg_name}\tNA\tNA\tNA")
                continue
            speedup = j.ms_per_step / m.ms_per_step
            print(f"{neuron}\t{cfg_name}\t{j.ms_per_step:.4f}\t{m.ms_per_step:.4f}\t{speedup:.2f}x")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"{item['backend']} {item['neuron']} {item['config']}: {item['error']}")

    print(f"\nWrote: {args.json}")


if __name__ == "__main__":
    main()
