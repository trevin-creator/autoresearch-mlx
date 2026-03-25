from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from world_model_experiments.autopilot_bridge import ArbitrationDecision, MotorAutopilotBridge
from world_model_experiments.command_interface import MotorCommandPacket
from world_model_experiments.fallback_controller import ConservativeFallbackController


class FlightMode(str, Enum):
    SHADOW = "shadow"
    AUTONOMOUS = "autonomous"
    FALLBACK = "fallback"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class FlightPlanTarget:
    relative_position: np.ndarray
    relative_yaw: float
    absolute_position: np.ndarray
    absolute_yaw: float


class FlightPlanCodec:
    block_dim = 8

    @classmethod
    def decode(cls, flight_plan: np.ndarray | None) -> list[FlightPlanTarget]:
        if flight_plan is None:
            return []
        arr = np.asarray(flight_plan, dtype=np.float32).reshape(-1)
        usable = (arr.shape[0] // cls.block_dim) * cls.block_dim
        if usable == 0:
            return []
        blocks = arr[:usable].reshape(-1, cls.block_dim)
        targets: list[FlightPlanTarget] = []
        for block in blocks:
            targets.append(
                FlightPlanTarget(
                    relative_position=block[:3].astype(np.float32),
                    relative_yaw=float(block[3]),
                    absolute_position=block[4:7].astype(np.float32),
                    absolute_yaw=float(block[7]),
                )
            )
        return targets

    @classmethod
    def primary(cls, flight_plan: np.ndarray | None) -> FlightPlanTarget | None:
        targets = cls.decode(flight_plan)
        return targets[0] if targets else None


@dataclass(frozen=True)
class FlightStateMachineConfig:
    initial_mode: str = "shadow"
    shadow_warmup_steps: int = 4
    fallback_hold_steps: int = 2
    recovery_hold_steps: int = 3
    max_tracking_error: float = 5.0
    recovery_ood_scale: float = 0.5


@dataclass(frozen=True)
class FlightStateSnapshot:
    mode: str
    mode_changed: float
    transition_reason: str
    source: str
    target_available: float
    tracking_error: float
    used_fallback: float
    shield_emergency: float


class FlightStateMachine:
    def __init__(
        self,
        bridge: MotorAutopilotBridge,
        fallback: ConservativeFallbackController,
        cfg: FlightStateMachineConfig,
    ) -> None:
        self.bridge = bridge
        self.fallback = fallback
        self.cfg = cfg
        self.mode = FlightMode(cfg.initial_mode)
        self._shadow_healthy_steps = 0
        self._fallback_bad_steps = 0
        self._recovery_good_steps = 0

    def reset(self) -> None:
        self.bridge.reset()
        self.mode = FlightMode(self.cfg.initial_mode)
        self._shadow_healthy_steps = 0
        self._fallback_bad_steps = 0
        self._recovery_good_steps = 0

    @staticmethod
    def _tracking_error(pose: np.ndarray, pose_delta: np.ndarray, target: FlightPlanTarget | None) -> float:
        if target is None:
            return 0.0
        pose = np.asarray(pose, dtype=np.float32)
        pose_delta = np.asarray(pose_delta, dtype=np.float32)
        abs_err = float(np.linalg.norm(target.absolute_position - pose[:3]))
        rel_err = float(np.linalg.norm(target.relative_position - pose_delta[:3]))
        yaw_err = float(abs(target.absolute_yaw - float(pose[5])))
        rel_yaw_err = float(abs(target.relative_yaw - float(pose_delta[5])))
        return abs_err + 0.5 * rel_err + 0.25 * yaw_err + 0.25 * rel_yaw_err

    def _set_mode(self, next_mode: FlightMode, reason: str) -> tuple[float, str]:
        if next_mode == self.mode:
            return 0.0, ""
        self.mode = next_mode
        if next_mode == FlightMode.SHADOW:
            self._shadow_healthy_steps = 0
            self._fallback_bad_steps = 0
            self._recovery_good_steps = 0
        elif next_mode == FlightMode.AUTONOMOUS:
            self._fallback_bad_steps = 0
            self._recovery_good_steps = 0
        elif next_mode == FlightMode.FALLBACK:
            self._recovery_good_steps = 0
        return 1.0, reason

    def step(
        self,
        timestamp_us: int,
        planner_command: np.ndarray,
        pose: np.ndarray,
        pose_delta: np.ndarray,
        flight_plan: np.ndarray | None,
        state_norm: float,
        ood_score: float,
    ) -> tuple[MotorCommandPacket, ArbitrationDecision, FlightStateSnapshot]:
        target = FlightPlanCodec.primary(flight_plan)
        tracking_error = self._tracking_error(pose, pose_delta, target)
        target_available = 1.0 if target is not None else 0.0

        mode_changed = 0.0
        transition_reason = ""
        healthy = ood_score <= self.bridge.arbitration_cfg.ood_fallback_threshold
        recovered = ood_score <= self.bridge.arbitration_cfg.ood_fallback_threshold * self.cfg.recovery_ood_scale
        diverged = tracking_error > self.cfg.max_tracking_error

        if self.mode == FlightMode.SHADOW:
            self._shadow_healthy_steps = (self._shadow_healthy_steps + 1) if healthy else 0
            if self._shadow_healthy_steps >= self.cfg.shadow_warmup_steps:
                mode_changed, transition_reason = self._set_mode(FlightMode.AUTONOMOUS, "shadow_warmup_complete")
        elif self.mode == FlightMode.AUTONOMOUS:
            self._fallback_bad_steps = (self._fallback_bad_steps + 1) if (not healthy or diverged) else 0
            if self._fallback_bad_steps >= self.cfg.fallback_hold_steps:
                mode_changed, transition_reason = self._set_mode(FlightMode.FALLBACK, "planner_unhealthy")
        elif self.mode == FlightMode.FALLBACK:
            self._recovery_good_steps = (self._recovery_good_steps + 1) if (recovered and not diverged) else 0
            if self._recovery_good_steps >= self.cfg.recovery_hold_steps:
                mode_changed, transition_reason = self._set_mode(FlightMode.SHADOW, "fallback_recovered")

        fallback_mode = "land" if self.mode in {FlightMode.FALLBACK, FlightMode.EMERGENCY_STOP} else "hover"
        fallback_command = self.fallback.command(pose=pose, pose_delta=pose_delta, mode=fallback_mode)

        effective_ood = float(ood_score)
        planner = np.asarray(planner_command, dtype=np.float32)
        if self.mode == FlightMode.SHADOW:
            effective_ood = self.bridge.arbitration_cfg.ood_fallback_threshold + 1.0
        elif self.mode == FlightMode.FALLBACK:
            effective_ood = self.bridge.arbitration_cfg.ood_fallback_threshold + 1.0
        elif self.mode == FlightMode.EMERGENCY_STOP:
            planner = np.zeros(4, dtype=np.float32)
            fallback_command = np.zeros(4, dtype=np.float32)
            state_norm = max(state_norm, self.bridge.shield_cfg.emergency_stop_norm + 1.0)
            effective_ood = self.bridge.arbitration_cfg.ood_fallback_threshold + 1.0

        packet, decision = self.bridge.step(
            timestamp_us=timestamp_us,
            planner_command=planner,
            fallback_command=fallback_command,
            state_norm=state_norm,
            ood_score=effective_ood,
        )

        if decision.shield_emergency > 0.0 and self.mode != FlightMode.EMERGENCY_STOP:
            mode_changed, transition_reason = self._set_mode(FlightMode.EMERGENCY_STOP, "shield_emergency")

        snapshot = FlightStateSnapshot(
            mode=self.mode.value,
            mode_changed=mode_changed,
            transition_reason=transition_reason,
            source=decision.source,
            target_available=target_available,
            tracking_error=float(tracking_error),
            used_fallback=decision.used_fallback,
            shield_emergency=decision.shield_emergency,
        )
        return packet, decision, snapshot