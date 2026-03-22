from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
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
    n_timed_runs: int
    mean_seconds: float
    median_seconds: float
    p10_seconds: float
    p90_seconds: float
    std_seconds: float
    coeff_var: float
    steps_per_second: float
    ms_per_step: float


@dataclass
class ParityCheckResult:
    neuron: str
    ok: bool
    trials: int
    atol: float
    rtol: float
    max_abs_out: float
    max_abs_state: float
    worst_trial: int
    message: str


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


def _time_jax(
    neuron: str,
    batch: int,
    hidden: int,
    steps: int,
    repeats: int,
    warmup_repeats: int,
):
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

    for _ in range(warmup_repeats):
        y, s = fn(params, xs, s0)
        jax.block_until_ready(y)
        jax.block_until_ready(s)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y, s = fn(params, xs, s0)
        jax.block_until_ready(y)
        jax.block_until_ready(s)
        times.append(time.perf_counter() - t0)
    return np.array(times, dtype=np.float64)


def _time_mlx(
    neuron: str,
    batch: int,
    hidden: int,
    steps: int,
    repeats: int,
    warmup_repeats: int,
):
    rng = np.random.default_rng(0)
    xs_np = rng.standard_normal((steps, batch, hidden), dtype=np.float32)
    xs = mx.array(xs_np)
    s0 = _state_for(neuron, batch, hidden, xp="mlx")
    fn = _mlx_runner(neuron, hidden)

    y, s = fn(xs, s0)
    mx.eval(y, s)

    for _ in range(warmup_repeats):
        y, s = fn(xs, s0)
        mx.eval(y, s)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y, s = fn(xs, s0)
        mx.eval(y, s)
        times.append(time.perf_counter() - t0)
    return np.array(times, dtype=np.float64)


def _replace_param_named(tree, name: str, value):
    if isinstance(tree, dict):
        updated = {}
        for key, node in tree.items():
            if key == name:
                updated[key] = value
            else:
                updated[key] = _replace_param_named(node, name, value)
        return updated
    return tree


def _forward_jax(
    neuron: str,
    xs_np: np.ndarray,
    s0_np: np.ndarray,
    w_rec_np: np.ndarray | None = None,
):
    xs = jnp.array(xs_np)
    s0 = jnp.array(s0_np)
    transformed = _spyx_transformed(neuron, xs_np.shape[-1])
    params = transformed.init(jax.random.PRNGKey(0), xs, s0)
    if w_rec_np is not None:
        params = _replace_param_named(
            params, "w", jnp.array(w_rec_np, dtype=jnp.float32)
        )
    apply_fn = jax.jit(transformed.apply)
    y, s = apply_fn(params, xs, s0)
    y = np.array(jax.block_until_ready(y))
    s = np.array(jax.block_until_ready(s))
    return y, s


def _forward_mlx(
    neuron: str,
    xs_np: np.ndarray,
    s0_np: np.ndarray,
    w_rec_np: np.ndarray | None = None,
):
    xs = mx.array(xs_np)
    s0 = mx.array(s0_np)
    hidden = xs_np.shape[-1]
    if neuron == "RLIF":
        cell = mlx_nn.RLIF(hidden_shape=(hidden,), beta_init=0.9, threshold=1.0)
        if w_rec_np is not None:
            cell.w_rec = mx.array(w_rec_np)

        def fn(x_in, s_in):
            outs = []
            state = s_in
            for t in range(x_in.shape[0]):
                out, state = cell(x_in[t], state)
                outs.append(out)
            return mx.stack(outs, axis=0), state

    else:
        fn = _mlx_runner(neuron, hidden)

    y, s = fn(xs, s0)
    mx.eval(y, s)
    return np.array(y), np.array(s)


def _parity_check_neuron(
    neuron: str,
    batch: int,
    hidden: int,
    steps: int,
    trials: int,
    base_seed: int,
    atol: float,
    rtol: float,
) -> ParityCheckResult:
    worst_out = 0.0
    worst_state = 0.0
    worst_trial = -1
    ok = True
    msg = "pass"

    for trial in range(trials):
        rng = np.random.default_rng(base_seed + trial)
        xs_np = rng.standard_normal((steps, batch, hidden), dtype=np.float32)
        s0_np = np.array(_state_for(neuron, batch, hidden, xp="jax"), dtype=np.float32)

        w_rec_np = None
        if neuron == "RLIF":
            w_rec_np = rng.standard_normal((hidden, hidden), dtype=np.float32) * 0.2

        try:
            y_jax, s_jax = _forward_jax(neuron, xs_np, s0_np, w_rec_np=w_rec_np)
            y_mlx, s_mlx = _forward_mlx(neuron, xs_np, s0_np, w_rec_np=w_rec_np)
        except Exception as exc:  # noqa: BLE001
            return ParityCheckResult(
                neuron=neuron,
                ok=False,
                trials=trials,
                atol=atol,
                rtol=rtol,
                max_abs_out=float("inf"),
                max_abs_state=float("inf"),
                worst_trial=trial,
                message=f"Execution error: {type(exc).__name__}: {exc}",
            )

        abs_out = float(np.max(np.abs(y_jax - y_mlx)))
        abs_state = float(np.max(np.abs(s_jax - s_mlx)))
        if abs_out > worst_out or abs_state > worst_state:
            worst_out = max(worst_out, abs_out)
            worst_state = max(worst_state, abs_state)
            worst_trial = trial

        out_ok = np.allclose(y_jax, y_mlx, atol=atol, rtol=rtol)
        state_ok = np.allclose(s_jax, s_mlx, atol=atol, rtol=rtol)
        if not (out_ok and state_ok):
            ok = False
            msg = "mismatch"
            break

    return ParityCheckResult(
        neuron=neuron,
        ok=ok,
        trials=trials,
        atol=atol,
        rtol=rtol,
        max_abs_out=worst_out,
        max_abs_state=worst_state,
        worst_trial=worst_trial,
        message=msg,
    )


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
    median_s = float(np.median(times))
    p10_s = float(np.percentile(times, 10))
    p90_s = float(np.percentile(times, 90))
    std_s = float(times.std())
    cv = float(std_s / mean_s) if mean_s > 0 else 0.0
    return BenchResult(
        neuron=neuron,
        config=cfg_name,
        batch=batch,
        hidden=hidden,
        steps=steps,
        backend=backend,
        n_timed_runs=int(times.size),
        mean_seconds=mean_s,
        median_seconds=median_s,
        p10_seconds=p10_s,
        p90_seconds=p90_s,
        std_seconds=std_s,
        coeff_var=cv,
        steps_per_second=float(steps / mean_s),
        ms_per_step=float(mean_s * 1000.0 / steps),
    )


def _run_parity_gate(run_parity_checks: bool) -> tuple[bool, str]:
    if not run_parity_checks:
        return True, "Parity gate disabled by CLI flag."

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "spyx/tests/test_networks.py",
        "spyx/tests/test_mlx_strict_parity.py",
        "spyx/tests/test_jax_mlx_direct_parity.py",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT), check=False
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode == 0:
        return True, output
    return False, output


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark spyx (JAX) vs spyx_mlx (MLX)"
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--json", type=Path, default=Path("bench_spyx_vs_mlx.json"))
    parser.add_argument("--mlx-device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument(
        "--parity-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run parity tests before benchmarking and abort if they fail.",
    )
    parser.add_argument(
        "--auto-skip-incompatible-jax-metal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If JAX Metal stack is likely incompatible, skip JAX timing and continue with MLX.",
    )
    parser.add_argument(
        "--bench-parity-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run per-neuron JAX-vs-MLX forward parity checks before timing.",
    )
    parser.add_argument(
        "--allow-bench-on-parity-fail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue timing neurons even if benchmark parity check fails.",
    )
    parser.add_argument(
        "--fail-on-bench-parity-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero if any benchmark parity precheck mismatches.",
    )
    parser.add_argument("--bench-parity-batch", type=int, default=8)
    parser.add_argument("--bench-parity-hidden", type=int, default=64)
    parser.add_argument("--bench-parity-steps", type=int, default=16)
    parser.add_argument("--bench-parity-trials", type=int, default=3)
    parser.add_argument("--bench-parity-seed", type=int, default=123)
    parser.add_argument("--bench-parity-atol", type=float, default=1e-5)
    parser.add_argument("--bench-parity-rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.mlx_device == "cpu":
        mx.set_default_device(mx.cpu)
    elif args.mlx_device == "gpu":
        mx.set_default_device(mx.gpu)

    parity_ok, parity_output = _run_parity_gate(args.parity_gate)
    if not parity_ok:
        print("Parity gate failed. Refusing to run benchmark timings.")
        print(parity_output)
        payload = {
            "run_config": {
                "repeats": args.repeats,
                "mlx_device": args.mlx_device,
                "skip_jax": args.skip_jax,
                "skip_mlx": args.skip_mlx,
                "parity_gate": args.parity_gate,
            },
            "results": [],
            "failures": [
                {
                    "backend": "parity-gate",
                    "neuron": "ALL",
                    "config": "preflight",
                    "error": "Parity tests failed; see parity_gate_output.",
                }
            ],
            "parity_gate_output": parity_output,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        raise SystemExit(2)

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
    parity_results: list[ParityCheckResult] = []
    blocked_neurons: set[str] = set()
    bench_parity_mismatch = False

    if args.bench_parity_check and not effective_skip_jax and not args.skip_mlx:
        print("\nRunning benchmark parity prechecks...")
        for neuron in neurons:
            check = _parity_check_neuron(
                neuron=neuron,
                batch=args.bench_parity_batch,
                hidden=args.bench_parity_hidden,
                steps=args.bench_parity_steps,
                trials=args.bench_parity_trials,
                base_seed=args.bench_parity_seed,
                atol=args.bench_parity_atol,
                rtol=args.bench_parity_rtol,
            )
            parity_results.append(check)
            print(
                f"parity {neuron}: {check.message} "
                f"(trials={check.trials}, max_abs_out={check.max_abs_out:.3e}, "
                f"max_abs_state={check.max_abs_state:.3e}, worst_trial={check.worst_trial})"
            )
            if not check.ok:
                bench_parity_mismatch = True
                failures.append(
                    {
                        "backend": "bench-parity",
                        "neuron": neuron,
                        "config": "precheck",
                        "error": (
                            f"Parity mismatch at atol={check.atol}, rtol={check.rtol}; "
                            f"max_abs_out={check.max_abs_out:.3e}, max_abs_state={check.max_abs_state:.3e}, "
                            f"worst_trial={check.worst_trial}"
                        ),
                    }
                )
                if not args.allow_bench_on_parity_fail:
                    blocked_neurons.add(neuron)

    for neuron in neurons:
        if neuron in blocked_neurons:
            print(f"skip timings for {neuron}: benchmark parity precheck failed")
            continue
        for cfg_name, batch, hidden, steps in configs:
            try:
                if not effective_skip_jax:
                    jax_times = _time_jax(
                        neuron,
                        batch,
                        hidden,
                        steps,
                        args.repeats,
                        args.warmup_repeats,
                    )
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
                    mlx_times = _time_mlx(
                        neuron,
                        batch,
                        hidden,
                        steps,
                        args.repeats,
                        args.warmup_repeats,
                    )
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
            "warmup_repeats": args.warmup_repeats,
            "mlx_device": args.mlx_device,
            "skip_jax": effective_skip_jax,
            "skip_mlx": args.skip_mlx,
            "parity_gate": args.parity_gate,
            "bench_parity_check": args.bench_parity_check,
            "allow_bench_on_parity_fail": args.allow_bench_on_parity_fail,
            "fail_on_bench_parity_mismatch": args.fail_on_bench_parity_mismatch,
            "bench_parity_batch": args.bench_parity_batch,
            "bench_parity_hidden": args.bench_parity_hidden,
            "bench_parity_steps": args.bench_parity_steps,
            "bench_parity_trials": args.bench_parity_trials,
            "bench_parity_seed": args.bench_parity_seed,
            "bench_parity_atol": args.bench_parity_atol,
            "bench_parity_rtol": args.bench_parity_rtol,
            "jax_backend": jax.default_backend(),
            "mlx_default_device": str(mx.default_device()),
            "jax_stack": jax_stack,
            "compat_warning": compat_warning,
        },
        "results": [asdict(r) for r in rows],
        "failures": failures,
        "parity_gate_output": parity_output,
        "bench_parity_results": [asdict(r) for r in parity_results],
        "bench_parity_blocked_neurons": sorted(blocked_neurons),
        "bench_parity_mismatch": bench_parity_mismatch,
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

    if args.fail_on_bench_parity_mismatch and bench_parity_mismatch:
        print("Hard-fail enabled: benchmark parity mismatches were detected.")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
