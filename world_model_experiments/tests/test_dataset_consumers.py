from __future__ import annotations

from pathlib import Path

import numpy as np

from world_model_experiments._io import write_sequence_dataset
from world_model_experiments.train_feature_lewm import FeatureSequenceDataset
from world_model_experiments.train_informed_dreamer import InformedDataset


def _sample_arrays() -> dict[str, np.ndarray]:
    return {
        "features": np.arange(48, dtype=np.float32).reshape(2, 3, 8),
        "actions": np.ones((2, 3, 4), dtype=np.float32),
        "motor_commands": np.full((2, 3, 4), 2.0, dtype=np.float32),
        "pose": np.zeros((2, 3, 6), dtype=np.float32),
        "pose_delta": np.full((2, 3, 6), 0.25, dtype=np.float32),
        "reward": np.full((2, 3), 1.5, dtype=np.float32),
        "continue": np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        "flight_plan": np.full((2, 3, 8), 3.0, dtype=np.float32),
        "timestamps_us": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64),
    }


def test_feature_sequence_dataset_reads_parquet(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    write_sequence_dataset(path, _sample_arrays())

    dataset = FeatureSequenceDataset(path, use_motor_commands=False)

    assert len(dataset) == 2
    sample = dataset[0]
    assert tuple(sample["features"].shape) == (3, 8)
    assert tuple(sample["actions"].shape) == (3, 4)
    assert tuple(sample["flight_plan"].shape) == (3, 8)


def test_informed_dataset_reads_parquet_with_motor_commands(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    write_sequence_dataset(path, _sample_arrays())

    dataset = InformedDataset(path, use_flight_plan=False, use_motor_commands=True)

    assert len(dataset) == 2
    sample = dataset[0]
    assert tuple(sample["features"].shape) == (3, 8)
    assert tuple(sample["actions"].shape) == (3, 4)
    assert tuple(sample["pose_delta"].shape) == (3, 6)
