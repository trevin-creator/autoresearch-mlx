from __future__ import annotations

import h5py
import numpy as np
import pytest

from world_model_experiments._errors import ERR_NO_MOTOR_COMMANDS
from world_model_experiments._io import load_actions


def _write_dataset(path: str) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("actions", data=np.ones((2, 3, 4), dtype=np.float32))
        h5.create_dataset("motor_commands", data=np.full((2, 3, 4), 2.0, dtype=np.float32))
        h5.create_dataset("flight_plan", data=np.full((2, 3, 8), 3.0, dtype=np.float32))


def test_load_actions_uses_motor_commands_when_requested(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "sample.h5"
    _write_dataset(str(path))
    with h5py.File(path, "r") as h5:
        loaded = load_actions(h5, use_motor_commands=True, use_flight_plan=False)
    assert loaded.shape == (2, 3, 4)
    assert np.allclose(loaded, 2.0)


def test_load_actions_concatenates_flight_plan_when_enabled(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "sample.h5"
    _write_dataset(str(path))
    with h5py.File(path, "r") as h5:
        loaded = load_actions(h5, use_motor_commands=False, use_flight_plan=True)
    assert loaded.shape == (2, 3, 12)
    assert np.allclose(loaded[..., :4], 1.0)
    assert np.allclose(loaded[..., 4:], 3.0)


def test_load_actions_raises_if_motor_commands_missing(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "missing_motor.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("actions", data=np.ones((1, 1, 4), dtype=np.float32))

    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match=ERR_NO_MOTOR_COMMANDS):
        load_actions(h5, use_motor_commands=True, use_flight_plan=False)
