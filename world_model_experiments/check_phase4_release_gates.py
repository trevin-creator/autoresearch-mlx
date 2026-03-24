from __future__ import annotations

import argparse
import ast
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Release-grade gate checks for phase 4")
    p.add_argument("--closed-loop-log", type=str, required=True)
    p.add_argument("--robust-log", type=str, required=True)
    p.add_argument("--runtime-log", type=str, required=True)
    p.add_argument("--onnx-log", type=str, required=True)
    p.add_argument("--replay-log", type=str, required=True)
    p.add_argument("--shadow-log", type=str, required=True)
    p.add_argument("--fault-log", type=str, required=True)
    p.add_argument("--sync-log", type=str, required=True)
    p.add_argument("--ood-log", type=str, required=True)
    p.add_argument("--system-id-log", type=str, required=True)
    p.add_argument("--manifest-path", type=str, required=True)

    p.add_argument("--max-crash-rate", type=float, default=0.01)
    p.add_argument("--max-termination-rate", type=float, default=0.01)
    p.add_argument("--min-survival", type=float, default=0.95)
    p.add_argument("--max-latency-p95-ms", type=float, default=6.0)
    p.add_argument("--max-parity-diff", type=float, default=1e-4)
    p.add_argument("--max-replay-pose-delta-mse", type=float, default=0.01)
    p.add_argument("--max-replay-reward-mse", type=float, default=0.02)
    p.add_argument("--max-shadow-fallback-rate", type=float, default=0.5)
    p.add_argument("--max-shadow-emergency-rate", type=float, default=0.01)
    p.add_argument("--min-shadow-autonomous-rate", type=float, default=0.5)
    p.add_argument("--max-shadow-emergency-stop-rate", type=float, default=0.01)
    p.add_argument("--min-fault-fallback-rate", type=float, default=0.6)
    p.add_argument("--max-fault-emergency-stop-rate", type=float, default=0.01)
    p.add_argument("--max-fault-shield-emergency-rate", type=float, default=0.01)
    p.add_argument("--max-shield-emergency-rate", type=float, default=0.01)
    p.add_argument("--max-sync-jitter-us", type=float, default=5000.0)
    p.add_argument("--max-ood-rate", type=float, default=0.10)
    p.add_argument("--min-system-id-corr", type=float, default=0.05)
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
    raise TypeError(f"Cannot convert to float: {value!r}")


def _parse_dict(lines: list[str], prefix: str) -> dict[str, float]:
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


def _parse_json(lines: list[str], prefix: str) -> dict[str, dict[str, float]]:
    for line in reversed(lines):
        if line.startswith(prefix + " "):
            payload = line[len(prefix) + 1 :].strip()
            parsed = json.loads(payload)
            out: dict[str, dict[str, float]] = {}
            if isinstance(parsed, dict):
                parsed_obj = cast(dict[str, object], parsed)
                for k, v in parsed_obj.items():
                    if isinstance(v, dict):
                        v_obj = cast(dict[str, object], v)
                        out[str(k)] = {str(mk): _to_float(mv) for mk, mv in v_obj.items()}
            return out
    raise RuntimeError(f"Could not parse {prefix}")


def _fail_if(cond: bool, msg: str, failures: list[str]) -> None:
    if cond:
        failures.append(msg)


def _load_manifest(path: str) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, JSONDecodeError):
        return {}
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    return {}


def main() -> None:
    args = parse_args()

    closed = _parse_dict(_read_lines(args.closed_loop_log), "closed_loop_eval")
    robust = _parse_json(_read_lines(args.robust_log), "robust_eval")
    runtime = _parse_dict(_read_lines(args.runtime_log), "runtime_benchmark")
    onnx = _parse_dict(_read_lines(args.onnx_log), "onnx_parity")
    replay = _parse_dict(_read_lines(args.replay_log), "real_replay_eval")
    shadow = _parse_dict(_read_lines(args.shadow_log), "shadow_mode_eval")
    fault = _parse_dict(_read_lines(args.fault_log), "fault_replay_eval")
    sync = _parse_dict(_read_lines(args.sync_log), "sensor_sync")
    ood = _parse_dict(_read_lines(args.ood_log), "ood_guard")
    sid = _parse_dict(_read_lines(args.system_id_log), "system_id")
    manifest = _load_manifest(args.manifest_path)

    failures: list[str] = []

    _fail_if(closed.get("crash_rate", 1.0) > args.max_crash_rate, "closed-loop crash rate too high", failures)
    _fail_if(
        closed.get("termination_rate", 1.0) > args.max_termination_rate,
        "closed-loop termination rate too high",
        failures,
    )
    _fail_if(closed.get("rollout_survival_mean", 0.0) < args.min_survival, "closed-loop survival too low", failures)
    _fail_if(
        closed.get("shield_emergency_rate", 1.0) > args.max_shield_emergency_rate,
        "closed-loop shield emergency rate too high",
        failures,
    )

    for name, sc in robust.items():
        _fail_if(sc.get("crash_rate", 1.0) > args.max_crash_rate, f"robust {name} crash rate too high", failures)
        _fail_if(
            sc.get("termination_rate", 1.0) > args.max_termination_rate,
            f"robust {name} termination rate too high",
            failures,
        )
        _fail_if(sc.get("rollout_survival_mean", 0.0) < args.min_survival, f"robust {name} survival too low", failures)
        _fail_if(
            sc.get("shield_emergency_rate", 1.0) > args.max_shield_emergency_rate,
            f"robust {name} shield emergency too high",
            failures,
        )

    _fail_if(runtime.get("latency_ms_p95", 1e9) > args.max_latency_p95_ms, "runtime p95 too high", failures)
    _fail_if(runtime.get("parity_max_abs_mean", 1e9) > args.max_parity_diff, "runtime parity diff too high", failures)
    _fail_if(onnx.get("max_abs_diff", 1e9) > args.max_parity_diff, "onnx parity diff too high", failures)

    _fail_if(
        replay.get("global_pose_delta_mse", 1e9) > args.max_replay_pose_delta_mse,
        "replay pose_delta mse too high",
        failures,
    )
    _fail_if(replay.get("global_reward_mse", 1e9) > args.max_replay_reward_mse, "replay reward mse too high", failures)
    _fail_if(shadow.get("fallback_rate", 1e9) > args.max_shadow_fallback_rate, "shadow fallback rate too high", failures)
    _fail_if(
        shadow.get("shield_emergency_rate", 1e9) > args.max_shadow_emergency_rate,
        "shadow shield emergency rate too high",
        failures,
    )
    _fail_if(
        shadow.get("autonomous_rate", -1.0) < args.min_shadow_autonomous_rate,
        "shadow autonomous rate too low",
        failures,
    )
    _fail_if(
        shadow.get("emergency_stop_rate", 1e9) > args.max_shadow_emergency_stop_rate,
        "shadow emergency-stop rate too high",
        failures,
    )
    _fail_if(
        fault.get("fallback_rate", 0.0) < args.min_fault_fallback_rate,
        "fault replay fallback rate too low",
        failures,
    )
    _fail_if(
        fault.get("emergency_stop_rate", 1e9) > args.max_fault_emergency_stop_rate,
        "fault replay emergency-stop rate too high",
        failures,
    )
    _fail_if(
        fault.get("shield_emergency_rate", 1e9) > args.max_fault_shield_emergency_rate,
        "fault replay shield emergency rate too high",
        failures,
    )
    _fail_if(sync.get("pass", 0.0) < 1.0, "sensor sync failed", failures)
    _fail_if(sync.get("jitter_p95_us", 1e9) > args.max_sync_jitter_us, "sensor sync jitter too high", failures)
    _fail_if(ood.get("pass", 0.0) < 1.0, "ood guard failed", failures)
    _fail_if(ood.get("frame_ood_rate", 1e9) > args.max_ood_rate, "ood rate too high", failures)
    _fail_if(sid.get("corr_velocity", -1.0) < args.min_system_id_corr, "system-id corr_velocity too low", failures)

    model_block = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    ckpt_hash = model_block.get("checkpoint_sha256") if isinstance(model_block, dict) else None
    onnx_hash = model_block.get("onnx_sha256") if isinstance(model_block, dict) else None
    _fail_if(not bool(ckpt_hash), "deployment manifest missing checkpoint hash", failures)
    _fail_if(not bool(onnx_hash), "deployment manifest missing onnx hash", failures)

    thresholds = manifest.get("gate_thresholds") if isinstance(manifest.get("gate_thresholds"), dict) else {}
    _fail_if(not isinstance(thresholds, dict), "deployment manifest missing gate_thresholds", failures)
    if isinstance(thresholds, dict):
        _fail_if(
            abs(float(thresholds.get("min_fault_fallback_rate", -1.0)) - args.min_fault_fallback_rate) > 1e-9,
            "deployment manifest threshold mismatch: min_fault_fallback_rate",
            failures,
        )
        _fail_if(
            abs(float(thresholds.get("max_fault_emergency_stop_rate", -1.0)) - args.max_fault_emergency_stop_rate) > 1e-9,
            "deployment manifest threshold mismatch: max_fault_emergency_stop_rate",
            failures,
        )

    if failures:
        print("phase4_release_gates FAILED")
        for f in failures:
            print("-", f)
        raise SystemExit(1)

    print("phase4_release_gates PASSED")


if __name__ == "__main__":
    main()
