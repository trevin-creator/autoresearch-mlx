from __future__ import annotations

import numpy as np

from world_model_experiments.safety_shield import SafetyShieldConfig, apply_safety_shield


def test_apply_safety_shield_clips_bounds_and_marks_modified() -> None:
    cfg = SafetyShieldConfig(motor_low=-1.0, motor_high=1.0, max_slew=1.0, max_abs_cmd=1.0, emergency_stop_norm=30.0)
    cmd, reasons = apply_safety_shield(
        command=np.array([2.0, -2.0, 0.0, 0.0], dtype=np.float32),
        prev_command=None,
        state_norm=0.0,
        cfg=cfg,
    )
    assert np.all(cmd <= 1.0)
    assert np.all(cmd >= -1.0)
    assert reasons["clipped_bounds"] == 1.0
    assert reasons["modified"] == 1.0


def test_apply_safety_shield_limits_slew_against_previous_command() -> None:
    cfg = SafetyShieldConfig(max_slew=0.2, max_abs_cmd=1.0, emergency_stop_norm=30.0)
    prev = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cmd, reasons = apply_safety_shield(
        command=np.array([0.8, -0.8, 0.4, -0.4], dtype=np.float32),
        prev_command=prev,
        state_norm=0.0,
        cfg=cfg,
    )
    assert np.allclose(cmd, np.array([0.2, -0.2, 0.2, -0.2], dtype=np.float32))
    assert reasons["slew_limited"] == 1.0
    assert reasons["modified"] == 1.0


def test_apply_safety_shield_emergency_stop_zeroes_command() -> None:
    cfg = SafetyShieldConfig(emergency_stop_norm=1.0)
    cmd, reasons = apply_safety_shield(
        command=np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        prev_command=None,
        state_norm=10.0,
        cfg=cfg,
    )
    assert np.allclose(cmd, np.zeros(4, dtype=np.float32))
    assert reasons["emergency_stop"] == 1.0
    assert reasons["modified"] == 1.0
