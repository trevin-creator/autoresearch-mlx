from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyShieldConfig:
    motor_low: float = -1.0
    motor_high: float = 1.0
    max_slew: float = 0.2
    max_abs_cmd: float = 1.0
    emergency_stop_norm: float = 30.0


def apply_safety_shield(
    command: np.ndarray,
    prev_command: np.ndarray | None,
    state_norm: float,
    cfg: SafetyShieldConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    cmd = np.asarray(command, dtype=np.float32).copy()

    reasons = {
        "clipped_bounds": 0.0,
        "slew_limited": 0.0,
        "abs_limited": 0.0,
        "emergency_stop": 0.0,
        "modified": 0.0,
    }

    clipped = np.clip(cmd, cfg.motor_low, cfg.motor_high)
    if not np.allclose(clipped, cmd):
        reasons["clipped_bounds"] = 1.0
    cmd = clipped

    if prev_command is not None and cfg.max_slew > 0.0:
        prev = np.asarray(prev_command, dtype=np.float32)
        delta = np.clip(cmd - prev, -cfg.max_slew, cfg.max_slew)
        slew_cmd = prev + delta
        if not np.allclose(slew_cmd, cmd):
            reasons["slew_limited"] = 1.0
        cmd = slew_cmd

    if cfg.max_abs_cmd > 0.0:
        abs_cmd = np.clip(cmd, -cfg.max_abs_cmd, cfg.max_abs_cmd)
        if not np.allclose(abs_cmd, cmd):
            reasons["abs_limited"] = 1.0
        cmd = abs_cmd

    if state_norm > cfg.emergency_stop_norm:
        cmd = np.zeros_like(cmd)
        reasons["emergency_stop"] = 1.0

    if any(v > 0.0 for k, v in reasons.items() if k != "modified"):
        reasons["modified"] = 1.0

    return cmd.astype(np.float32), reasons
