from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model_experiments._io import load_sequence_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit simple motor-command to response dynamics from rollout dataset")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output", type=str, default="artifacts/sim/system_id/motor_dynamics.json")
    p.add_argument("--max-delay", type=int, default=4)
    return p.parse_args()


def _estimate_delay(cmd: np.ndarray, response: np.ndarray, max_delay: int) -> int:
    best_d = 0
    best_c = -1.0
    for d in range(max_delay + 1):
        if d >= len(cmd) - 2:
            break
        c = np.corrcoef(cmd[: -d or None], response[d:])[0, 1]
        c = 0.0 if np.isnan(c) else float(c)
        if c > best_c:
            best_c = c
            best_d = d
    return best_d


def main() -> None:
    args = parse_args()

    dataset = load_sequence_dataset(args.dataset)
    if "motor_commands" not in dataset or "pose_delta" not in dataset:
        raise ValueError("dataset must include motor_commands and pose_delta")  # noqa: TRY003
    cmd = np.asarray(dataset["motor_commands"], dtype=np.float64)
    pose_delta = np.asarray(dataset["pose_delta"], dtype=np.float64)

    u = cmd.reshape(-1, cmd.shape[-1])
    y = pose_delta.reshape(-1, pose_delta.shape[-1])

    u_mean = np.mean(u, axis=-1)
    v_response = np.linalg.norm(y[:, :3], axis=-1)
    yaw_response = np.abs(y[:, 5])

    delay_v = _estimate_delay(u_mean, v_response, args.max_delay)
    delay_yaw = _estimate_delay(u_mean, yaw_response, args.max_delay)

    d = max(delay_v, delay_yaw)
    u_aligned = u_mean[: -d or None]
    v_aligned = v_response[d:]
    y_aligned = yaw_response[d:]

    # Linear fits as coarse system-ID proxies.
    kv = float(np.dot(u_aligned, v_aligned) / (np.dot(u_aligned, u_aligned) + 1e-9))
    kyaw = float(np.dot(u_aligned, y_aligned) / (np.dot(u_aligned, u_aligned) + 1e-9))

    result = {
        "delay_steps": float(d),
        "gain_velocity": kv,
        "gain_yaw": kyaw,
        "corr_velocity": float(np.corrcoef(u_aligned, v_aligned)[0, 1]),
        "corr_yaw": float(np.corrcoef(u_aligned, y_aligned)[0, 1]),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    print("system_id", result)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
