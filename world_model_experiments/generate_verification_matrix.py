from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate requirements-traceable verification matrix markdown")
    p.add_argument("--closed-loop-log", type=str, required=True)
    p.add_argument("--robust-log", type=str, required=True)
    p.add_argument("--runtime-log", type=str, required=True)
    p.add_argument("--replay-log", type=str, required=True)
    p.add_argument("--shadow-log", type=str, required=True)
    p.add_argument("--fault-log", type=str, required=True)
    p.add_argument("--manifest-path", type=str, required=True)
    p.add_argument("--sync-log", type=str, required=True)
    p.add_argument("--ood-log", type=str, required=True)
    p.add_argument("--system-id-log", type=str, required=True)
    p.add_argument("--output", type=str, default="artifacts/sim/verification/phase5_verification_matrix.md")
    return p.parse_args()


def _read(path: str) -> list[str]:
    return Path(path).read_text().splitlines()


def _parse_dict(lines: list[str], prefix: str) -> dict[str, float]:
    for line in reversed(lines):
        if line.startswith(prefix + " "):
            payload = line[len(prefix) + 1 :].strip()
            parsed = ast.literal_eval(payload)
            if isinstance(parsed, dict):
                return {str(k): float(v) for k, v in parsed.items()}
    raise RuntimeError(f"Missing {prefix}")


def _parse_json(lines: list[str], prefix: str) -> dict[str, dict[str, float]]:
    for line in reversed(lines):
        if line.startswith(prefix + " "):
            payload = line[len(prefix) + 1 :].strip()
            raw = json.loads(payload)
            out: dict[str, dict[str, float]] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict):
                        out[str(k)] = {str(mk): float(mv) for mk, mv in v.items()}
            return out
    raise RuntimeError(f"Missing {prefix}")


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _read_json(path: str) -> dict[str, object]:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        return raw
    return {}


def main() -> None:
    args = parse_args()
    closed = _parse_dict(_read(args.closed_loop_log), "closed_loop_eval")
    robust = _parse_json(_read(args.robust_log), "robust_eval")
    runtime = _parse_dict(_read(args.runtime_log), "runtime_benchmark")
    replay = _parse_dict(_read(args.replay_log), "real_replay_eval")
    shadow = _parse_dict(_read(args.shadow_log), "shadow_mode_eval")
    fault = _parse_dict(_read(args.fault_log), "fault_replay_eval")
    manifest = _read_json(args.manifest_path)
    model_block = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    has_ckpt_hash = bool(model_block.get("checkpoint_sha256")) if isinstance(model_block, dict) else False
    has_onnx_hash = bool(model_block.get("onnx_sha256")) if isinstance(model_block, dict) else False
    sync = _parse_dict(_read(args.sync_log), "sensor_sync")
    ood = _parse_dict(_read(args.ood_log), "ood_guard")
    sid = _parse_dict(_read(args.system_id_log), "system_id")

    robust_crash_max = max((v.get("crash_rate", 0.0) for v in robust.values()), default=0.0)

    rows = [
        ("R1 Closed-loop safety", _status(closed.get("crash_rate", 1.0) <= 0.01), f"crash_rate={closed.get('crash_rate', -1):.6f}"),
        (
            "R2 Robustness safety",
            _status(robust_crash_max <= 0.01),
            f"max_robust_crash_rate={robust_crash_max:.6f}",
        ),
        (
            "R3 Runtime budget",
            _status(runtime.get("latency_ms_p95", 1e9) <= 6.0),
            f"latency_ms_p95={runtime.get('latency_ms_p95', -1):.6f}",
        ),
        (
            "R4 Replay parity",
            _status(replay.get("global_pose_delta_mse", 1e9) <= 0.01),
            f"pose_delta_mse={replay.get('global_pose_delta_mse', -1):.6f}",
        ),
        (
            "R5 Sensor sync",
            _status(sync.get("pass", 0.0) >= 1.0),
            f"jitter_p95_us={sync.get('jitter_p95_us', -1):.3f}",
        ),
        (
            "R6 OOD guard",
            _status(ood.get("pass", 0.0) >= 1.0),
            f"frame_ood_rate={ood.get('frame_ood_rate', -1):.6f}",
        ),
        (
            "R7 System identification",
            _status(sid.get("corr_velocity", 0.0) >= 0.05),
            f"corr_velocity={sid.get('corr_velocity', -1):.6f}",
        ),
        (
            "R8 Shadow arbitration",
            _status(shadow.get("shield_emergency_rate", 1.0) <= 0.01),
            (
                f"fallback_rate={shadow.get('fallback_rate', -1):.6f}, "
                f"shield_emergency_rate={shadow.get('shield_emergency_rate', -1):.6f}"
            ),
        ),
        (
            "R9 Flight-mode progression",
            _status(
                shadow.get("autonomous_rate", 0.0) >= 0.5
                and shadow.get("emergency_stop_rate", 1.0) <= 0.01
            ),
            (
                f"autonomous_rate={shadow.get('autonomous_rate', -1):.6f}, "
                f"emergency_stop_rate={shadow.get('emergency_stop_rate', -1):.6f}"
            ),
        ),
        (
            "R10 Fault-arbitration resilience",
            _status(fault.get("fallback_rate", 0.0) >= 0.6 and fault.get("emergency_stop_rate", 1.0) <= 0.01),
            f"fault_fallback_rate={fault.get('fallback_rate', -1):.6f}, fault_emergency_stop_rate={fault.get('emergency_stop_rate', -1):.6f}",
        ),
        (
            "R11 Deployment manifest pinned",
            _status(has_ckpt_hash and has_onnx_hash),
            f"version_tag={manifest.get('version_tag', 'unknown')}, release_gate_passed={manifest.get('release_gate_passed', False)}",
        ),
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 5 Verification Matrix", "", "| Requirement | Status | Evidence |", "|---|---|---|"]
    for req, st, ev in rows:
        lines.append(f"| {req} | {st} | {ev} |")

    out.write_text("\n".join(lines) + "\n")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
