from __future__ import annotations

from pathlib import Path

import numpy as np

from world_model_experiments._io import write_sequence_dataset
from world_model_experiments.data_catalog import load_dataset_record_for_path
from world_model_experiments.materialize_dataset import materialize_source


def _sample_arrays() -> dict[str, np.ndarray]:
    return {
        "features": np.arange(12, dtype=np.float32).reshape(1, 3, 4),
        "actions": np.ones((1, 3, 2), dtype=np.float32),
        "pose": np.zeros((1, 3, 6), dtype=np.float32),
        "pose_delta": np.full((1, 3, 6), 0.5, dtype=np.float32),
        "reward": np.full((1, 3), 1.0, dtype=np.float32),
        "continue": np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        "timestamps_us": np.array([[10, 20, 30]], dtype=np.int64),
    }


def test_materialize_source_from_local_file(tmp_path: Path) -> None:
    src = tmp_path / "source.parquet"
    dst = tmp_path / "materialized.parquet"
    write_sequence_dataset(src, _sample_arrays())

    out = materialize_source(source_uri=str(src), destination=dst)

    assert out == dst
    assert dst.exists()


def test_materialized_file_can_be_registered(tmp_path: Path) -> None:
    from world_model_experiments.data_catalog import register_dataset

    src = tmp_path / "source.parquet"
    dst = tmp_path / "materialized.parquet"
    write_sequence_dataset(src, _sample_arrays())
    materialize_source(source_uri=str(src), destination=dst)

    record = register_dataset(
        dataset_path=dst,
        dataset_name="materialized_test",
        transform_name="materialize_dataset",
        source_uri=str(src),
        metadata={"kind": "unit-test"},
        schema={"features": {"shape": [1, 3, 4], "dtype": "float32"}},
        catalog_root=tmp_path / "catalog",
    )
    loaded = load_dataset_record_for_path(dst)
    assert loaded is not None
    assert loaded.dataset_id == record.dataset_id
