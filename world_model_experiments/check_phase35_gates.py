from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fail-fast gate checks for phase-3.5 metrics")
    p.add_argument("--closed-loop-log", type=str, required=True)
    p.add_argument("--robust-log", type=str, required=True)
    p.add_argument("--runtime-log", type=str, required=True)
    p.add_argument("--onnx-log", type=str, required=True)

    p.add_argument("--max-crash-rate", type=float, default=0.05)
    p.add_argument("--max-termination-rate", type=float, default=0.05)
    p.add_argument("--min-survival", type=float, default=0.90)
    p.add_argument("--max-latency-p95-ms", type=float, default=5.0)
    p.add_argument("--max-parity-diff", type=float, default=1e-4)
    return p.parse_args()


def _read_lines(path: str) -> list[str]:
    return Path(path).read_text().splitlines()


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Cannot convert value to float: {value!r}")


def _parse_prefixed_dict(lines: list[str], prefix: str) -> dict[str, float]:
    for line in reversed(lines):
        if line.startswith(prefix + " "):
            payload = line[len(prefix) + 1 :].strip()
            parsed = ast.literal_eval(payload)
            if isinstance(parsed, dict):
                parsed_obj = cast(dict[str, object], parsed)
                out: dict[str, float] = {}
                for k, v in parsed_obj.items():
                    out[str(k)] = _to_float(v)
                return out
    raise RuntimeError(f"Could not parse {prefix}")


def _parse_prefixed_json(lines: list[str], prefix: str) -> dict[str, dict[str, float]]:
    for line in reversed(lines):
        if line.startswith(prefix + " "):
            payload = line[len(prefix) + 1 :].strip()
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise RuntimeError(f"Parsed payload for {prefix} is not an object")
            parsed_obj = cast(dict[str, object], parsed)
            out: dict[str, dict[str, float]] = {}
            for scenario, metrics_any in parsed_obj.items():
                if not isinstance(metrics_any, dict):
                    continue
                metrics_obj = cast(dict[str, object], metrics_any)
                metrics: dict[str, float] = {}
                for k, v in metrics_obj.items():
                    metrics[str(k)] = _to_float(v)
                out[str(scenario)] = metrics
            return out
    raise RuntimeError(f"Could not parse {prefix}")


def _parse_prefixed_json_flat(lines: list[str], prefix: str) -> dict[str, float]:
    for line in reversed(lines):
        if line.startswith(prefix + " "):
            payload = line[len(prefix) + 1 :].strip()
            parsed_any: Any = json.loads(payload)
            if not isinstance(parsed_any, dict):
                raise RuntimeError(f"Parsed payload for {prefix} is not an object")
            parsed_obj = cast(dict[str, object], parsed_any)
            out: dict[str, float] = {}
            for k, v in parsed_obj.items():
                out[str(k)] = _to_float(v)
            return out
    raise RuntimeError(f"Could not parse {prefix}")


def main() -> None:
    args = parse_args()

    closed = _parse_prefixed_dict(_read_lines(args.closed_loop_log), "closed_loop_eval")
    robust = _parse_prefixed_json(_read_lines(args.robust_log), "robust_eval")
    runtime = _parse_prefixed_json_flat(_read_lines(args.runtime_log), "runtime_benchmark")
    onnx = _parse_prefixed_dict(_read_lines(args.onnx_log), "onnx_parity")

    failures: list[str] = []

    if closed.get("crash_rate", 1.0) > args.max_crash_rate:
        failures.append(f"closed_loop crash_rate {closed.get('crash_rate')} > {args.max_crash_rate}")
    if closed.get("termination_rate", 1.0) > args.max_termination_rate:
        failures.append(
            f"closed_loop termination_rate {closed.get('termination_rate')} > {args.max_termination_rate}"
        )
    if closed.get("rollout_survival_mean", 0.0) < args.min_survival:
        failures.append(f"closed_loop rollout_survival_mean {closed.get('rollout_survival_mean')} < {args.min_survival}")

    for scenario, metrics in robust.items():
        if metrics.get("crash_rate", 1.0) > args.max_crash_rate:
            failures.append(f"robust {scenario} crash_rate {metrics.get('crash_rate')} > {args.max_crash_rate}")
        if metrics.get("termination_rate", 1.0) > args.max_termination_rate:
            failures.append(
                f"robust {scenario} termination_rate {metrics.get('termination_rate')} > {args.max_termination_rate}"
            )
        if metrics.get("rollout_survival_mean", 0.0) < args.min_survival:
            failures.append(
                f"robust {scenario} rollout_survival_mean {metrics.get('rollout_survival_mean')} < {args.min_survival}"
            )

    if runtime.get("latency_ms_p95", 1e9) > args.max_latency_p95_ms:
        failures.append(f"runtime latency_ms_p95 {runtime.get('latency_ms_p95')} > {args.max_latency_p95_ms}")
    if runtime.get("parity_max_abs_mean", 1e9) > args.max_parity_diff:
        failures.append(f"runtime parity_max_abs_mean {runtime.get('parity_max_abs_mean')} > {args.max_parity_diff}")

    if onnx.get("max_abs_diff", 1e9) > args.max_parity_diff:
        failures.append(f"onnx max_abs_diff {onnx.get('max_abs_diff')} > {args.max_parity_diff}")

    if failures:
        print("phase35_gates FAILED")
        for f in failures:
            print("-", f)
        raise SystemExit(1)

    print("phase35_gates PASSED")


if __name__ == "__main__":
    main()
