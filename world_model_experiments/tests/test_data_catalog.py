from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from world_model_experiments._io import load_sequence_dataset, write_sequence_dataset
from world_model_experiments.data_catalog import load_catalog, load_dataset_record_for_path, register_dataset


def _sample_arrays() -> dict[str, np.ndarray]:
    return {
        "features": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "actions": np.ones((2, 3, 2), dtype=np.float32),
        "pose": np.zeros((2, 3, 6), dtype=np.float32),
        "pose_delta": np.full((2, 3, 6), 0.5, dtype=np.float32),
        "reward": np.full((2, 3), 1.5, dtype=np.float32),
        "continue": np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        "timestamps_us": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64),
    }


def test_write_and_load_parquet_sequence_dataset(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "sample.parquet"
    arrays = _sample_arrays()

    write_sequence_dataset(path, arrays)
    loaded = load_sequence_dataset(path)

    assert loaded["features"].shape == (2, 3, 4)
    assert loaded["timestamps_us"].dtype == np.int64
    assert np.allclose(loaded["features"], arrays["features"])
    assert np.allclose(loaded["reward"], arrays["reward"])


def test_register_dataset_writes_manifest_and_catalog(tmp_path: pytest.TempPathFactory) -> None:
    dataset_path = tmp_path / "managed.parquet"
    catalog_root = tmp_path / "catalog"
    arrays = _sample_arrays()
    write_sequence_dataset(dataset_path, arrays)

    record = register_dataset(
        dataset_path=dataset_path,
        dataset_name="demo_dataset",
        transform_name="unit_test_builder",
        split="train",
        source_uri="synthetic://unit-test",
        parent_ids=["raw:seed0"],
        metadata={"seed": 7},
        schema={name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        catalog_root=catalog_root,
    )

    manifest = load_dataset_record_for_path(dataset_path)
    assert manifest is not None
    assert manifest.dataset_id == record.dataset_id
    assert manifest.source_uri == "synthetic://unit-test"
    assert manifest.parent_ids == ("raw:seed0",)
    assert json.loads(manifest.metadata_json)["seed"] == 7

    catalog = load_catalog(catalog_root)
    assert len(catalog) == 1
    assert catalog[0].dataset_id == record.dataset_id
    assert catalog[0].row_count == 2
