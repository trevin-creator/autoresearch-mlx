from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from world_model_experiments.data_catalog import DatasetRecord, load_dataset_record_for_path

DEFAULT_MLFLOW_TRACKING_DIR = Path("artifacts/mlruns")


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        msg = "mlflow is required for experiment tracking; install world_model_experiments/requirements.txt"
        raise RuntimeError(msg) from exc
    return mlflow


def configure_mlflow(experiment_name: str, tracking_uri: str | Path | None = None) -> Any:
    mlflow = _require_mlflow()
    root = Path(tracking_uri) if tracking_uri is not None else DEFAULT_MLFLOW_TRACKING_DIR
    mlflow.set_tracking_uri(str(root.resolve()))
    mlflow.set_experiment(experiment_name)
    return mlflow


def resolve_dataset_record(dataset_path: str | Path) -> DatasetRecord | None:
    return load_dataset_record_for_path(dataset_path)


def dataset_tags(record: DatasetRecord) -> dict[str, str]:
    tags = {
        "dataset.id": record.dataset_id,
        "dataset.name": record.dataset_name,
        "dataset.version": record.version,
        "dataset.transform": record.transform_name,
        "dataset.format": record.storage_format,
        "dataset.row_count": str(-1 if record.row_count is None else record.row_count),
        "dataset.parent_count": str(len(record.parent_ids)),
    }
    if record.source_uri:
        tags["dataset.source_uri"] = record.source_uri
    if record.parent_ids:
        tags["dataset.parent_ids"] = ",".join(record.parent_ids)
    metadata = record.metadata()
    if "split" in metadata:
        tags["dataset.meta.split"] = str(metadata["split"])
    if "seed" in metadata:
        tags["dataset.meta.seed"] = str(metadata["seed"])
    return tags


def log_dataset_record(mlflow: Any, record: DatasetRecord | None) -> None:
    if record is None:
        return
    mlflow.set_tags(dataset_tags(record))
    schema = record.schema()
    schema_keys = sorted(schema.keys()) if isinstance(schema, dict) else []
    metadata = record.metadata()
    mlflow.log_params(
        {
            "dataset_name": record.dataset_name,
            "dataset_version": record.version,
            "dataset_format": record.storage_format,
            "dataset_row_count": -1 if record.row_count is None else record.row_count,
            "dataset_path": record.path,
            "dataset_transform": record.transform_name,
            "dataset_parent_count": len(record.parent_ids),
            "dataset_schema_keys": ",".join(schema_keys),
            "dataset_metadata_keys": ",".join(sorted(metadata.keys())) if isinstance(metadata, dict) else "",
        }
    )
    manifest_path = Path(record.manifest_path)
    if manifest_path.exists():
        mlflow.log_artifact(str(manifest_path), artifact_path="dataset")


def log_flat_params(mlflow: Any, params: Mapping[str, Any]) -> None:
    normalized = {
        key: str(value) if isinstance(value, (list, tuple, dict, set)) else value for key, value in params.items()
    }
    mlflow.log_params(normalized)
