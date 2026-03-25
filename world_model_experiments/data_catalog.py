from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CATALOG_ROOT = Path("artifacts/data_catalog")
DEFAULT_DATASET_ROOT = Path("artifacts/datasets")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def default_dataset_output_path(dataset_name: str, suffix: str = ".parquet") -> Path:
    return DEFAULT_DATASET_ROOT / dataset_name / utc_stamp() / f"data{suffix}"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for parquet-backed dataset lineage; install world_model_experiments/requirements.txt"
        raise RuntimeError(msg) from exc
    return pa, pq


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def describe_arrays(arrays: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        schema[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    return schema


def _row_count_from_schema(schema: Mapping[str, Mapping[str, Any]]) -> int | None:
    for value in schema.values():
        shape = value.get("shape")
        if isinstance(shape, list) and shape:
            try:
                return int(shape[0])
            except (TypeError, ValueError):
                return None
    return None


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    dataset_name: str
    version: str
    path: str
    manifest_path: str
    storage_format: str
    created_utc: str
    transform_name: str
    split: str
    source_uri: str | None
    parent_ids: tuple[str, ...]
    row_count: int | None
    sha256: str
    metadata_json: str
    schema_json: str

    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    def schema(self) -> dict[str, Any]:
        return json.loads(self.schema_json)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "path": self.path,
            "storage_format": self.storage_format,
            "created_utc": self.created_utc,
            "transform_name": self.transform_name,
            "split": self.split,
            "source_uri": self.source_uri,
            "parent_ids": list(self.parent_ids),
            "row_count": self.row_count,
            "sha256": self.sha256,
            "metadata": self.metadata(),
            "schema": self.schema(),
        }

    def to_catalog_row(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "path": self.path,
            "manifest_path": self.manifest_path,
            "storage_format": self.storage_format,
            "created_utc": self.created_utc,
            "transform_name": self.transform_name,
            "split": self.split,
            "source_uri": self.source_uri,
            "parent_ids_json": _json_dumps(list(self.parent_ids)),
            "row_count": self.row_count,
            "sha256": self.sha256,
            "metadata_json": self.metadata_json,
            "schema_json": self.schema_json,
        }

    @classmethod
    def from_catalog_row(cls, row: Mapping[str, Any]) -> DatasetRecord:
        return cls(
            dataset_id=str(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]),
            version=str(row["version"]),
            path=str(row["path"]),
            manifest_path=str(row["manifest_path"]),
            storage_format=str(row["storage_format"]),
            created_utc=str(row["created_utc"]),
            transform_name=str(row["transform_name"]),
            split=str(row["split"]),
            source_uri=None if row.get("source_uri") in (None, "") else str(row["source_uri"]),
            parent_ids=tuple(json.loads(str(row.get("parent_ids_json", "[]")))),
            row_count=None if row.get("row_count") is None else int(row["row_count"]),
            sha256=str(row["sha256"]),
            metadata_json=str(row["metadata_json"]),
            schema_json=str(row["schema_json"]),
        )

    @classmethod
    def from_manifest_dict(cls, manifest: Mapping[str, Any], manifest_path: str | Path) -> DatasetRecord:
        return cls(
            dataset_id=str(manifest["dataset_id"]),
            dataset_name=str(manifest["dataset_name"]),
            version=str(manifest["version"]),
            path=str(manifest["path"]),
            manifest_path=str(manifest_path),
            storage_format=str(manifest["storage_format"]),
            created_utc=str(manifest["created_utc"]),
            transform_name=str(manifest["transform_name"]),
            split=str(manifest["split"]),
            source_uri=None if manifest.get("source_uri") is None else str(manifest["source_uri"]),
            parent_ids=tuple(str(item) for item in manifest.get("parent_ids", [])),
            row_count=None if manifest.get("row_count") is None else int(manifest["row_count"]),
            sha256=str(manifest["sha256"]),
            metadata_json=_json_dumps(manifest.get("metadata", {})),
            schema_json=_json_dumps(manifest.get("schema", {})),
        )


def catalog_registry_path(catalog_root: str | Path = DEFAULT_CATALOG_ROOT) -> Path:
    return Path(catalog_root) / "dataset_registry.parquet"


def load_catalog(catalog_root: str | Path = DEFAULT_CATALOG_ROOT) -> list[DatasetRecord]:
    registry_path = catalog_registry_path(catalog_root)
    if not registry_path.exists():
        return []
    _, pq = _require_pyarrow()
    table = pq.read_table(registry_path)
    return [DatasetRecord.from_catalog_row(row) for row in table.to_pylist()]


def _write_catalog(records: Sequence[DatasetRecord], catalog_root: str | Path = DEFAULT_CATALOG_ROOT) -> Path:
    pa, pq = _require_pyarrow()
    registry_path = catalog_registry_path(catalog_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_catalog_row() for record in sorted(records, key=lambda item: (item.dataset_name, item.version))]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, registry_path, compression="zstd")
    return registry_path


def adjacent_manifest_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def load_dataset_record_for_path(dataset_path: str | Path) -> DatasetRecord | None:
    manifest_path = adjacent_manifest_path(dataset_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetRecord.from_manifest_dict(manifest, manifest_path)


def register_dataset(
    *,
    dataset_path: str | Path,
    dataset_name: str,
    transform_name: str,
    split: str = "train",
    source_uri: str | None = None,
    parent_ids: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    schema: Mapping[str, Any] | None = None,
    catalog_root: str | Path = DEFAULT_CATALOG_ROOT,
) -> DatasetRecord:
    path = Path(dataset_path).resolve()
    dataset_schema = dict(schema or {})
    created = utc_stamp()
    sha256 = file_sha256(path)
    version = f"{created}-{sha256[:8]}"
    manifest_path = adjacent_manifest_path(path)
    record = DatasetRecord(
        dataset_id=f"{dataset_name}:{version}",
        dataset_name=dataset_name,
        version=version,
        path=str(path),
        manifest_path=str(manifest_path),
        storage_format=path.suffix.lstrip(".").lower() or "unknown",
        created_utc=created,
        transform_name=transform_name,
        split=split,
        source_uri=source_uri,
        parent_ids=tuple(str(item) for item in parent_ids),
        row_count=_row_count_from_schema(dataset_schema),
        sha256=sha256,
        metadata_json=_json_dumps(dict(metadata or {})),
        schema_json=_json_dumps(dataset_schema),
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_json_dumps(record.to_manifest_dict()) + "\n", encoding="utf-8")

    existing = [item for item in load_catalog(catalog_root) if item.dataset_id != record.dataset_id]
    existing.append(record)
    _write_catalog(existing, catalog_root)
    return record
