from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_model_experiments.command_interface import CommandInterfaceConfig, MotorCommandInterface, MotorCommandPacket
from world_model_experiments.safety_shield import SafetyShieldConfig, apply_safety_shield


@dataclass(frozen=True)
class ArbitrationConfig:
    ood_fallback_threshold: float = 0.10
    shield_emergency_threshold: float = 0.0
    fallback_mode: str = "hover"


@dataclass(frozen=True)
class ArbitrationDecision:
    source: str
    used_fallback: float
    ood_score: float
    shield_modified: float
    shield_emergency: float


class MotorAutopilotBridge:
    def __init__(
        self,
        interface_cfg: CommandInterfaceConfig,
        shield_cfg: SafetyShieldConfig,
        arbitration_cfg: ArbitrationConfig,
    ) -> None:
        self.interface = MotorCommandInterface(interface_cfg)
        self.shield_cfg = shield_cfg
        self.arbitration_cfg = arbitration_cfg
        self.prev_command: np.ndarray | None = None

    def reset(self) -> None:
        self.interface.reset()
        self.prev_command = None

    def step(
        self,
        timestamp_us: int,
        planner_command: np.ndarray,
        fallback_command: np.ndarray,
        state_norm: float,
        ood_score: float,
    ) -> tuple[MotorCommandPacket, ArbitrationDecision]:
        use_fallback = float(ood_score > self.arbitration_cfg.ood_fallback_threshold)
        selected = np.asarray(fallback_command if use_fallback else planner_command, dtype=np.float32)
        source = "fallback" if use_fallback else "planner"

        safe_command, shield = apply_safety_shield(selected, self.prev_command, state_norm, self.shield_cfg)
        if shield["emergency_stop"] > self.arbitration_cfg.shield_emergency_threshold:
            source = "shield_emergency"

        packet = self.interface.encode(timestamp_us=timestamp_us, motor_norm=safe_command, source=source)
        self.prev_command = packet.motor_norm.copy()

        return packet, ArbitrationDecision(
            source=source,
            used_fallback=use_fallback,
            ood_score=float(ood_score),
            shield_modified=float(shield["modified"]),
            shield_emergency=float(shield["emergency_stop"]),
        )
