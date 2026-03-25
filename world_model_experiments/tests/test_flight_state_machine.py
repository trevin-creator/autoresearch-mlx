from __future__ import annotations

import numpy as np

from world_model_experiments.autopilot_bridge import ArbitrationConfig, MotorAutopilotBridge
from world_model_experiments.command_interface import CommandInterfaceConfig
from world_model_experiments.fallback_controller import ConservativeFallbackController, FallbackControllerConfig
from world_model_experiments.flight_state_machine import FlightStateMachine, FlightStateMachineConfig
from world_model_experiments.safety_shield import SafetyShieldConfig


def _build_state_machine(cfg: FlightStateMachineConfig) -> FlightStateMachine:
    bridge = MotorAutopilotBridge(
        interface_cfg=CommandInterfaceConfig(),
        shield_cfg=SafetyShieldConfig(emergency_stop_norm=10.0),
        arbitration_cfg=ArbitrationConfig(ood_fallback_threshold=0.1),
    )
    fallback = ConservativeFallbackController(FallbackControllerConfig())
    return FlightStateMachine(bridge=bridge, fallback=fallback, cfg=cfg)


def test_shadow_warmup_transitions_to_autonomous() -> None:
    sm = _build_state_machine(
        FlightStateMachineConfig(
            initial_mode="shadow",
            shadow_warmup_steps=2,
            fallback_hold_steps=1,
            recovery_hold_steps=1,
        )
    )

    pose = np.zeros(6, dtype=np.float32)
    pose_delta = np.zeros(6, dtype=np.float32)
    planner = np.zeros(4, dtype=np.float32)

    _, _, snap1 = sm.step(
        timestamp_us=0,
        planner_command=planner,
        pose=pose,
        pose_delta=pose_delta,
        flight_plan=None,
        state_norm=0.0,
        ood_score=0.01,
    )
    assert snap1.mode == "shadow"
    assert snap1.mode_changed == 0.0

    _, _, snap2 = sm.step(
        timestamp_us=1,
        planner_command=planner,
        pose=pose,
        pose_delta=pose_delta,
        flight_plan=None,
        state_norm=0.0,
        ood_score=0.01,
    )
    assert snap2.mode == "autonomous"
    assert snap2.mode_changed == 1.0
    assert snap2.transition_reason == "shadow_warmup_complete"


def test_shield_emergency_forces_emergency_stop_mode() -> None:
    sm = _build_state_machine(
        FlightStateMachineConfig(
            initial_mode="autonomous",
            shadow_warmup_steps=1,
            fallback_hold_steps=1,
            recovery_hold_steps=1,
        )
    )

    pose = np.zeros(6, dtype=np.float32)
    pose_delta = np.zeros(6, dtype=np.float32)
    planner = np.zeros(4, dtype=np.float32)

    _, decision, snap = sm.step(
        timestamp_us=0,
        planner_command=planner,
        pose=pose,
        pose_delta=pose_delta,
        flight_plan=None,
        state_norm=100.0,
        ood_score=0.0,
    )

    assert decision.shield_emergency == 1.0
    assert snap.mode == "emergency_stop"
    assert snap.mode_changed == 1.0
    assert snap.transition_reason == "shield_emergency"
