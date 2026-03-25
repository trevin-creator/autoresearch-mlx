"""Shared I/O helpers for world_model_experiments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from world_model_experiments._errors import ERR_NO_MOTOR_COMMANDS

_INT64_KEYS = {"timestamps_us"}


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for parquet datasets; install world_model_experiments/requirements.txt"
        raise RuntimeError(msg) from exc
    return pa, pq


def _coerce_column_dtype(name: str, values: list[Any]) -> np.ndarray:
    dtype = np.int64 if name in _INT64_KEYS else np.float32
    return np.asarray(values, dtype=dtype)


def load_sequence_dataset(path: str | Path) -> dict[str, np.ndarray]:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(dataset_path, "r") as h5:
            return {name: np.asarray(h5[name]) for name in h5}
    if suffix == ".parquet":
        _, pq = _require_pyarrow()
        table = pq.read_table(dataset_path)
        return {name: _coerce_column_dtype(name, table[name].to_pylist()) for name in table.column_names}
    raise ValueError(f"Unsupported dataset format: {dataset_path}")  # noqa: TRY003


def _write_hdf5_dataset(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as h5:
        for key, value in arrays.items():
            h5.create_dataset(key, data=np.asarray(value), compression="gzip")


def _write_parquet_dataset(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    pa, pq = _require_pyarrow()
    columns = {key: np.asarray(value).tolist() for key, value in arrays.items()}
    table = pa.Table.from_pydict(columns)
    pq.write_table(table, path, compression="zstd")


def write_sequence_dataset(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = dataset_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        _write_hdf5_dataset(dataset_path, arrays)
        return dataset_path
    if suffix == ".parquet":
        _write_parquet_dataset(dataset_path, arrays)
        return dataset_path
    raise ValueError(f"Unsupported dataset format: {dataset_path}")  # noqa: TRY003


def load_actions(dataset: Mapping[str, Any], use_motor_commands: bool, use_flight_plan: bool) -> np.ndarray:
    """Load action arrays from a dataset mapping, optionally concatenating flight plan."""
    if use_motor_commands:
        if "motor_commands" not in dataset:
            raise ValueError(ERR_NO_MOTOR_COMMANDS)
        actions = np.asarray(dataset["motor_commands"], dtype=np.float32)
    else:
        actions = np.asarray(dataset["actions"], dtype=np.float32)
    if use_flight_plan and "flight_plan" in dataset:
        actions = np.concatenate([actions, np.asarray(dataset["flight_plan"], dtype=np.float32)], axis=-1)
    return actions
