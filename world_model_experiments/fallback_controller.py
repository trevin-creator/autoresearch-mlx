from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FallbackControllerConfig:
    hover_throttle: float = 0.05
    roll_gain: float = 0.6
    pitch_gain: float = 0.6
    yaw_gain: float = 0.3
    vel_damp_gain: float = 0.2
    descend_bias: float = -0.05
    motor_limit: float = 0.4


class ConservativeFallbackController:
    """Simple conservative controller for shadow mode and emergency fallback.

    It produces a bounded, low-authority hover/descend style command intended to be
    safer than the learned planner when arbitration disables planner authority.
    """

    def __init__(self, cfg: FallbackControllerConfig) -> None:
        self.cfg = cfg

    def command(
        self,
        pose: np.ndarray,
        pose_delta: np.ndarray,
        mode: str = "hover",
    ) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32)
        pose_delta = np.asarray(pose_delta, dtype=np.float32)

        roll, pitch, yaw = float(pose[3]), float(pose[4]), float(pose[5])
        vz = float(pose_delta[2])
        yaw_rate = float(pose_delta[5])

        throttle = self.cfg.hover_throttle - self.cfg.vel_damp_gain * vz
        if mode == "land":
            throttle = throttle + self.cfg.descend_bias

        roll_term = -self.cfg.roll_gain * roll
        pitch_term = -self.cfg.pitch_gain * pitch
        yaw_term = -self.cfg.yaw_gain * (yaw + yaw_rate)

        # Motor order: [front_left, front_right, rear_right, rear_left]
        mix = np.asarray(
            [
                throttle + roll_term + pitch_term + yaw_term,
                throttle + roll_term - pitch_term - yaw_term,
                throttle - roll_term - pitch_term + yaw_term,
                throttle - roll_term + pitch_term - yaw_term,
            ],
            dtype=np.float32,
        )
        return np.clip(mix, -self.cfg.motor_limit, self.cfg.motor_limit)
