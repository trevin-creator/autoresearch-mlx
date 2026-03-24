from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a deployment manifest for motor-policy release artifacts")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--onnx-model", type=str, required=True)
    p.add_argument("--release-gates-log", type=str, required=True)
    p.add_argument("--shadow-log", type=str, required=True)
    p.add_argument("--fault-log", type=str, required=True)
    p.add_argument("--version-tag", type=str, default="dev")
    p.add_argument("--output", type=str, required=True)

    p.add_argument("--max-shadow-fallback-rate", type=float, default=0.5)
    p.add_argument("--max-shadow-emergency-rate", type=float, default=0.01)
    p.add_argument("--min-shadow-autonomous-rate", type=float, default=0.5)
    p.add_argument("--max-shadow-emergency-stop-rate", type=float, default=0.01)
    p.add_argument("--min-fault-fallback-rate", type=float, default=0.6)
    p.add_argument("--max-fault-emergency-stop-rate", type=float, default=0.01)
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tail_eval(path: Path, prefix: str) -> dict[str, float]:
    lines = path.read_text().splitlines()
    for line in reversed(lines):
        if line.startswith(prefix + " "):
            payload = line[len(prefix) + 1 :].strip()
            parsed = ast.literal_eval(payload)
            if isinstance(parsed, dict):
                out: dict[str, float] = {}
                for k, v in parsed.items():
                    if isinstance(v, (int, float, bool)):
                        out[str(k)] = float(v)
                return out
    return {}


def _release_gate_passed(path: Path) -> bool:
    lines = path.read_text().splitlines()
    return any("phase4_release_gates PASSED" in line for line in lines)


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    onnx_model = Path(args.onnx_model)
    release_log = Path(args.release_gates_log)
    shadow_log = Path(args.shadow_log)
    fault_log = Path(args.fault_log)
    out = Path(args.output)

    manifest = {
        "schema_version": "1.0",
        "version_tag": args.version_tag,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "release_gate_passed": _release_gate_passed(release_log),
        "model": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "onnx_path": str(onnx_model),
            "onnx_sha256": _sha256(onnx_model),
        },
        "gate_thresholds": {
            "max_shadow_fallback_rate": args.max_shadow_fallback_rate,
            "max_shadow_emergency_rate": args.max_shadow_emergency_rate,
            "min_shadow_autonomous_rate": args.min_shadow_autonomous_rate,
            "max_shadow_emergency_stop_rate": args.max_shadow_emergency_stop_rate,
            "min_fault_fallback_rate": args.min_fault_fallback_rate,
            "max_fault_emergency_stop_rate": args.max_fault_emergency_stop_rate,
        },
        "evidence": {
            "shadow_eval": _tail_eval(shadow_log, "shadow_mode_eval"),
            "fault_eval": _tail_eval(fault_log, "fault_replay_eval"),
            "release_log_path": str(release_log),
            "shadow_log_path": str(shadow_log),
            "fault_log_path": str(fault_log),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"deployment_manifest {out}")


if __name__ == "__main__":
    main()
