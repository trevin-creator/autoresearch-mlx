from __future__ import annotations

import argparse
import importlib.metadata
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

import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mlx.core as mx  # noqa: E402
import spyx.axn as spyx_axn  # noqa: E402
import spyx.nn as spyx_nn  # noqa: E402
import spyx_mlx.nn as mlx_nn  # noqa: E402


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


def _parse_semver(version: str) -> tuple[int, int, int]:
    core = version.split("+", 1)[0]
    parts = core.split(".")
    out = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 3:
        out.append(0)
    return out[0], out[1], out[2]


def _jax_stack_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "jax": None,
        "jaxlib": None,
        "jax-metal": None,
    }
    for pkg in info:
        try:
            info[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info[pkg] = None
    return info


def _metal_compat_warning(stack: dict[str, str | None]) -> str | None:
    jax_v = stack.get("jax")
    metal_v = stack.get("jax-metal")
    if jax.default_backend().lower() != "metal" or not jax_v or not metal_v:
        return None

    # Latest jax-metal stacks can fail with `default_memory_space` on some macOS setups.
    if _parse_semver(jax_v) >= (0, 5, 0):
        return (
            "Detected JAX Metal with jax>=0.5; this stack can raise "
            "`default_memory_space` during array creation/JIT on macOS. "
            "Recommended pinned benchmark stack: "
            "jax==0.4.26, jaxlib==0.4.26, jax-metal==0.1.0, dm-haiku==0.0.12, optax==0.2.2."
        )
    return None


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
        threshold = 1.0

        def model(xs, s0):
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(0.9))
            beta = jnp.clip(beta, 0.0, 1.0)
            spike_fn = spyx_axn.superspike()

            def step_fn(v, x_t):
                spikes = spike_fn(v - threshold)
                v = beta * v + x_t - spikes * threshold
                return v, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
    elif neuron == "ALIF":
        threshold = 1.0

        def model(xs, s0):
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(0.9))
            gamma = hk.get_parameter("gamma", [], init=hk.initializers.Constant(0.9))
            beta = jnp.clip(beta, 0.0, 1.0)
            gamma = jnp.clip(gamma, 0.0, 1.0)
            spike_fn = spyx_axn.superspike()

            def step_fn(vt, x_t):
                v, t = jnp.split(vt, 2, axis=-1)
                thresh = threshold + t
                spikes = spike_fn(v - thresh)
                v = beta * v + x_t - spikes * thresh
                t = gamma * t + (1.0 - gamma) * spikes
                vt = jnp.concatenate([v, t], axis=-1)
                return vt, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
    elif neuron == "CuBaLIF":
        threshold = 1.0

        def model(xs, s0):
            alpha = hk.get_parameter("alpha", [], init=hk.initializers.Constant(0.8))
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(0.9))
            alpha = jnp.clip(alpha, 0.0, 1.0)
            beta = jnp.clip(beta, 0.0, 1.0)
            spike_fn = spyx_axn.superspike()

            def step_fn(vi, x_t):
                v, i = jnp.split(vi, 2, axis=-1)
                spikes = spike_fn(v - threshold)
                reset = spikes * threshold
                v = v - reset
                i = alpha * i + x_t
                v = beta * v + i - reset
                vi = jnp.concatenate([v, i], axis=-1)
                return vi, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
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
        cell = mlx_nn.IF(hidden_shape=hidden_shape, threshold=1.0)
    elif neuron == "LIF":
        cell = mlx_nn.LIF(hidden_shape=hidden_shape, beta_init=0.9, threshold=1.0)
    elif neuron == "ALIF":
        cell = mlx_nn.ALIF(
            hidden_shape=hidden_shape, beta_init=0.9, gamma_init=0.9, threshold=1.0
        )
    elif neuron == "CuBaLIF":
        cell = mlx_nn.CuBaLIF(
            hidden_shape=hidden_shape, alpha_init=0.8, beta_init=0.9, threshold=1.0
        )
    elif neuron == "RLIF":
        cell = mlx_nn.RLIF(hidden_shape=hidden_shape, beta_init=0.9, threshold=1.0)
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
    except Exception as exc:
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


def _summary(
    neuron: str,
    cfg_name: str,
    batch: int,
    hidden: int,
    steps: int,
    backend: str,
    times: np.ndarray,
):
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
    parser = argparse.ArgumentParser(
        description="Benchmark spyx (JAX) vs spyx_mlx (MLX)"
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--json", type=Path, default=Path("bench_spyx_vs_mlx.json"))
    parser.add_argument("--mlx-device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument(
        "--auto-skip-incompatible-jax-metal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If JAX Metal stack is likely incompatible, skip JAX timing and continue with MLX.",
    )
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
    mlx_backend_name = (
        "spyx-mlx-gpu" if "gpu" in str(mx.default_device()).lower() else "spyx-mlx-cpu"
    )

    jax_stack = _jax_stack_info()
    compat_warning = _metal_compat_warning(jax_stack)
    effective_skip_jax = args.skip_jax
    if compat_warning:
        print(f"Compatibility warning: {compat_warning}")
        if args.auto_skip_incompatible_jax_metal and not args.skip_jax:
            effective_skip_jax = True
            print("Auto-skip enabled: skipping JAX timings for this run.")

    print(f"JAX backend: {jax.default_backend()}")
    print(f"MLX default device: {mx.default_device()}")

    rows: list[BenchResult] = []
    failures: list[dict[str, str]] = []
    for neuron in neurons:
        for cfg_name, batch, hidden, steps in configs:
            try:
                if not effective_skip_jax:
                    jax_times = _time_jax(neuron, batch, hidden, steps, args.repeats)
                    rows.append(
                        _summary(
                            neuron,
                            cfg_name,
                            batch,
                            hidden,
                            steps,
                            jax_backend_name,
                            jax_times,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "backend": jax_backend_name,
                        "neuron": neuron,
                        "config": cfg_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            try:
                if not args.skip_mlx:
                    mlx_times = _time_mlx(neuron, batch, hidden, steps, args.repeats)
                    rows.append(
                        _summary(
                            neuron,
                            cfg_name,
                            batch,
                            hidden,
                            steps,
                            mlx_backend_name,
                            mlx_times,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "backend": mlx_backend_name,
                        "neuron": neuron,
                        "config": cfg_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            print(f"done: {neuron} {cfg_name}")

    payload = {
        "run_config": {
            "repeats": args.repeats,
            "mlx_device": args.mlx_device,
            "skip_jax": effective_skip_jax,
            "skip_mlx": args.skip_mlx,
            "jax_backend": jax.default_backend(),
            "mlx_default_device": str(mx.default_device()),
            "jax_stack": jax_stack,
            "compat_warning": compat_warning,
        },
        "results": [asdict(r) for r in rows],
        "failures": failures,
    }
    args.json.write_text(json.dumps(payload, indent=2))

    print("\nSummary (ms/step, lower is better):")
    print(
        f"neuron\tconfig\t{jax_backend_name}\t{mlx_backend_name}\tspeedup(mlx_vs_jax)"
    )
    for neuron in neurons:
        for cfg_name, batch, hidden, steps in configs:
            j = next(
                (
                    r
                    for r in rows
                    if r.neuron == neuron
                    and r.config == cfg_name
                    and r.backend == jax_backend_name
                ),
                None,
            )
            m = next(
                (
                    r
                    for r in rows
                    if r.neuron == neuron
                    and r.config == cfg_name
                    and r.backend == mlx_backend_name
                ),
                None,
            )
            j_ms = f"{j.ms_per_step:.4f}" if j is not None else "NA"
            m_ms = f"{m.ms_per_step:.4f}" if m is not None else "NA"
            if j is None or m is None:
                print(f"{neuron}\t{cfg_name}\t{j_ms}\t{m_ms}\tNA")
                continue
            speedup = j.ms_per_step / m.ms_per_step
            print(f"{neuron}\t{cfg_name}\t{j_ms}\t{m_ms}\t{speedup:.2f}x")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(
                f"{item['backend']} {item['neuron']} {item['config']}: {item['error']}"
            )

    print(f"\nWrote: {args.json}")


if __name__ == "__main__":
    main()
