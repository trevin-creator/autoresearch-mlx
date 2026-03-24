from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommandInterfaceConfig:
    pwm_min: float = 1000.0
    pwm_max: float = 2000.0
    max_pwm_delta: float = 120.0
    neutral_pwm: float = 1500.0


@dataclass(frozen=True)
class MotorCommandPacket:
    timestamp_us: int
    motor_norm: np.ndarray  # shape [4], in [-1, 1]
    motor_pwm: np.ndarray  # shape [4], in [pwm_min, pwm_max]
    source: str


class MotorCommandInterface:
    """Convert normalized motor commands to bounded PWM commands with rate limiting."""

    def __init__(self, cfg: CommandInterfaceConfig):
        self.cfg = cfg
        self._last_pwm = np.full(4, cfg.neutral_pwm, dtype=np.float32)

    def reset(self) -> None:
        self._last_pwm = np.full(4, self.cfg.neutral_pwm, dtype=np.float32)

    def _norm_to_pwm(self, motor_norm: np.ndarray) -> np.ndarray:
        x = np.asarray(motor_norm, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        pwm = self.cfg.pwm_min + 0.5 * (x + 1.0) * (self.cfg.pwm_max - self.cfg.pwm_min)
        return np.clip(pwm, self.cfg.pwm_min, self.cfg.pwm_max)

    def encode(self, timestamp_us: int, motor_norm: np.ndarray, source: str = "planner") -> MotorCommandPacket:
        pwm = self._norm_to_pwm(motor_norm)
        delta = np.clip(pwm - self._last_pwm, -self.cfg.max_pwm_delta, self.cfg.max_pwm_delta)
        pwm = self._last_pwm + delta
        pwm = np.clip(pwm, self.cfg.pwm_min, self.cfg.pwm_max)
        self._last_pwm = pwm.astype(np.float32)

        return MotorCommandPacket(
            timestamp_us=int(timestamp_us),
            motor_norm=np.clip(np.asarray(motor_norm, dtype=np.float32), -1.0, 1.0),
            motor_pwm=self._last_pwm.copy(),
            source=source,
        )
