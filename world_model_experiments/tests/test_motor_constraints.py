from __future__ import annotations

import torch

from world_model_experiments.motor_constraints import apply_motor_constraints


def test_apply_motor_constraints_clamps_values() -> None:
    actions = torch.tensor([[2.0, -2.0, 0.5, -0.5]], dtype=torch.float32)
    out = apply_motor_constraints(actions, low=-1.0, high=1.0)
    assert out.shape == actions.shape
    assert torch.all(out <= 1.0)
    assert torch.all(out >= -1.0)


def test_apply_motor_constraints_limits_slew_along_time() -> None:
    actions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, -1.0],
                [1.0, -1.0],
            ]
        ],
        dtype=torch.float32,
    )

    out = apply_motor_constraints(actions, low=-1.0, high=1.0, max_delta=0.25)

    deltas = out[:, 1:, :] - out[:, :-1, :]
    assert torch.all(deltas <= 0.25 + 1e-6)
    assert torch.all(deltas >= -0.25 - 1e-6)


def test_apply_motor_constraints_preserves_2d_input_shape() -> None:
    actions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float32,
    )

    out = apply_motor_constraints(actions, low=-1.0, high=1.0, max_delta=0.5)
    assert out.shape == actions.shape
