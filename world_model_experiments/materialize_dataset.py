from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from world_model_experiments._io import load_sequence_dataset
from world_model_experiments.data_catalog import (
    DEFAULT_CATALOG_ROOT,
    default_dataset_output_path,
    describe_arrays,
    register_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Materialize a source dataset URI into managed storage with lineage registration"
    )
    p.add_argument("--source-uri", type=str, required=True)
    p.add_argument("--output", type=str, default=str(default_dataset_output_path("materialized_dataset")))
    p.add_argument("--dataset-name", type=str, default="materialized_dataset")
    p.add_argument("--transform-name", type=str, default="materialize_dataset")
    p.add_argument("--catalog-root", type=str, default=str(DEFAULT_CATALOG_ROOT))
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--parent-dataset-id", action="append", default=[])
    p.add_argument("--metadata-json", type=str, default="{}")
    p.add_argument("--hf-repo-type", type=str, default="dataset")
    p.add_argument("--hf-revision", type=str, default="main")
    return p.parse_args()


def _download_http(source_uri: str, destination: Path) -> None:
    with urllib.request.urlopen(source_uri) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_s3(source_uri: str, destination: Path) -> None:
    parsed = urlparse(source_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 source URI: {source_uri}")  # noqa: TRY003
    try:
        import boto3
    except ImportError as exc:
        msg = "boto3 is required for s3:// materialization; install world_model_experiments/requirements.txt"
        raise RuntimeError(msg) from exc
    client = boto3.client("s3")
    client.download_file(bucket, key, str(destination))


def _download_hf(source_uri: str, destination: Path, repo_type: str, revision: str) -> None:
    payload = source_uri[len("hf://") :]
    repo_id, sep, file_path = payload.partition("/")
    if not sep or not repo_id or not file_path:
        raise ValueError(  # noqa: TRY003
            "hf:// URI must be hf://<repo_id>/<path/in/repo>, for example hf://user/dataset/train.parquet"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        msg = "huggingface_hub is required for hf:// materialization; install world_model_experiments/requirements.txt"
        raise RuntimeError(msg) from exc
    cached_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type=repo_type, revision=revision)
    shutil.copy2(cached_path, destination)


def materialize_source(
    *,
    source_uri: str,
    destination: str | Path,
    hf_repo_type: str = "dataset",
    hf_revision: str = "main",
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if source_uri.startswith("s3://"):
        _download_s3(source_uri, destination_path)
        return destination_path
    if source_uri.startswith("hf://"):
        _download_hf(source_uri, destination_path, hf_repo_type, hf_revision)
        return destination_path
    if source_uri.startswith("http://") or source_uri.startswith("https://"):
        _download_http(source_uri, destination_path)
        return destination_path

    local_path = Path(source_uri[len("file://") :]) if source_uri.startswith("file://") else Path(source_uri)
    shutil.copy2(local_path, destination_path)
    return destination_path


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata_json)
    output = materialize_source(
        source_uri=args.source_uri,
        destination=args.output,
        hf_repo_type=args.hf_repo_type,
        hf_revision=args.hf_revision,
    )

    schema: dict[str, dict[str, object]] = {}
    try:
        arrays = load_sequence_dataset(output)
        schema = describe_arrays(arrays)
    except Exception:
        # Allow non-sequence files to still be tracked in lineage.
        schema = {}

    record = register_dataset(
        dataset_path=output,
        dataset_name=args.dataset_name,
        transform_name=args.transform_name,
        split=args.split,
        source_uri=args.source_uri,
        parent_ids=args.parent_dataset_id,
        metadata=metadata,
        schema=schema,
        catalog_root=args.catalog_root,
    )

    print(
        "materialized_dataset", {"path": str(output), "dataset_id": record.dataset_id, "manifest": record.manifest_path}
    )


if __name__ == "__main__":
    main()
