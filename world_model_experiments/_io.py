"""Shared I/O helpers for world_model_experiments."""

from __future__ import annotations

import h5py
import numpy as np

from world_model_experiments._errors import ERR_NO_MOTOR_COMMANDS


def load_actions(h5: h5py.File, use_motor_commands: bool, use_flight_plan: bool) -> np.ndarray:
    """Load action arrays from an HDF5 dataset, optionally concatenating flight plan."""
    if use_motor_commands:
        if "motor_commands" not in h5:
            raise ValueError(ERR_NO_MOTOR_COMMANDS)
        actions = np.asarray(h5["motor_commands"], dtype=np.float32)
    else:
        actions = np.asarray(h5["actions"], dtype=np.float32)
    if use_flight_plan and "flight_plan" in h5:
        actions = np.concatenate([actions, np.asarray(h5["flight_plan"], dtype=np.float32)], axis=-1)
    return actions
