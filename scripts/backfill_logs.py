#!/usr/bin/env python3
"""Best-effort historical reconstruction for MLX run logs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from log_utils import extract_short_commit_hints, parse_log_file

MATCH_RANK = {"none": 0, "weak": 1, "medium": 2, "strong": 3}
MATCH_TYPE_COUNTS = (
    "TSV-only",
    "latest-log enriched",
    "archived-log enriched",
    "reconstructed with low confidence",
)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=repo_root / "results.tsv")
    parser.add_argument("--logs-dir", type=Path, default=repo_root / "logs")
    parser.add_argument("--latest-log", type=Path, default=repo_root / "run.log")
    parser.add_argument("--current-commit", type=str, default=None)
    return parser.parse_args()


def read_results_rows(results_path: Path) -> list[dict[str, str]]:
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def resolve_current_commit(results_path: Path, explicit_commit: str | None) -> tuple[str | None, str]:
    if explicit_commit:
        return explicit_commit, "flag"

    repo_root = results_path.resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, "unresolved"
    return result.stdout.strip(), "git"


def make_base_row(row_index: int, row: dict[str, str]) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "commit": row["commit"],
        "val_bpb": row["val_bpb"],
        "status": row["status"],
        "matched_log_path": None,
        "match_type": "TSV-only",
        "match_strength": "none",
        "confidence": 0.0,
        "provenance_notes": "No archival or reconstructable log evidence found; row remains valid TSV-only evidence.",
    }


def log_summary_fields(parsed_log: dict[str, Any]) -> dict[str, Any] | None:
    summary = parsed_log.get("summary_fields", {})
    if not summary and not parsed_log.get("exists"):
        return None
    return {
        "is_final_summary": parsed_log.get("is_final_summary", False),
        "summary_fields": summary,
        "size_bytes": parsed_log.get("size_bytes", 0),
    }


def consider_candidate(best: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    best_rank = MATCH_RANK[best["match_strength"]]
    candidate_rank = MATCH_RANK[candidate["match_strength"]]
    if candidate_rank > best_rank:
        return candidate
    if candidate_rank == best_rank and candidate["confidence"] > best["confidence"]:
        return candidate
    return best


def val_matches(parsed_log: dict[str, Any], row_val_bpb: str) -> bool:
    parsed_val = parsed_log.get("summary_fields", {}).get("val_bpb")
    if parsed_val is None:
        return False
    return f"{parsed_val:.6f}" == row_val_bpb


def archive_candidate(row: dict[str, str], row_result: dict[str, Any], logs_dir: Path) -> dict[str, Any]:
    archive_path = logs_dir / f"{row['commit']}.log"
    parsed_log = parse_log_file(archive_path)
    if not parsed_log.get("exists"):
        return row_result

    note = f"Canonical archive matched at {archive_path} by commit."
    if val_matches(parsed_log, row["val_bpb"]):
        note += " Parsed val_bpb agrees with results.tsv."
    elif parsed_log["summary_fields"].get("val_bpb") is not None:
        note += " Parsed val_bpb differs from results.tsv, but commit match remains authoritative."
    elif not parsed_log["is_final_summary"]:
        note += " Log does not contain a full final summary, which is expected for some crash runs."

    candidate = {
        **row_result,
        "matched_log_path": str(archive_path.resolve()),
        "match_type": "archived-log enriched",
        "match_strength": "strong",
        "confidence": 1.0,
        "provenance_notes": note,
        "log_summary": log_summary_fields(parsed_log),
    }
    return consider_candidate(row_result, candidate)


def latest_log_candidate(
    row: dict[str, str],
    row_result: dict[str, Any],
    latest_log_path: Path,
    latest_log: dict[str, Any],
    current_commit: str | None,
    current_commit_source: str,
) -> dict[str, Any]:
    if current_commit is None or row["commit"] != current_commit or not latest_log.get("exists"):
        return row_result

    note = "Latest transient log matched using the resolved current commit."
    if current_commit_source == "flag":
        note += " Current commit came from --current-commit."
    elif current_commit_source == "git":
        note += " Current commit came from Git HEAD."
    if val_matches(latest_log, row["val_bpb"]):
        note += " Parsed val_bpb agrees with results.tsv."
    elif latest_log["summary_fields"].get("val_bpb") is not None:
        note += " Parsed val_bpb differs from results.tsv."
    elif not latest_log["is_final_summary"]:
        note += " Latest log is incomplete, so this remains transient evidence only."

    candidate = {
        **row_result,
        "matched_log_path": str(latest_log_path.resolve()),
        "match_type": "latest-log enriched",
        "match_strength": "medium",
        "confidence": 0.7,
        "provenance_notes": note,
        "log_summary": log_summary_fields(latest_log),
    }
    return consider_candidate(row_result, candidate)


def reconstruction_candidates(
    logs_dir: Path,
    canonical_commits: set[str],
) -> list[tuple[Path, dict[str, Any], list[str]]]:
    candidates: list[tuple[Path, dict[str, Any], list[str]]] = []
    if not logs_dir.exists():
        return candidates

    for log_path in sorted(logs_dir.glob("*.log")):
        stem = log_path.stem
        if stem in canonical_commits:
            continue
        parsed_log = parse_log_file(log_path)
        filename_hints = extract_short_commit_hints(log_path.name)
        candidates.append((log_path, parsed_log, filename_hints))
    return candidates


def weak_reconstruction(
    rows: list[dict[str, str]],
    row_results: list[dict[str, Any]],
    logs_dir: Path,
) -> list[dict[str, Any]]:
    by_commit = {row["commit"]: idx for idx, row in enumerate(rows)}
    unmatched_by_val: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        if row_results[idx]["match_strength"] != "none":
            continue
        unmatched_by_val.setdefault(row["val_bpb"], []).append(idx)

    used_paths: set[Path] = set()
    recon_logs = reconstruction_candidates(logs_dir, set(by_commit))
    for log_path, parsed_log, filename_hints in recon_logs:
        target_index: int | None = None
        note_bits = []

        hinted_indexes = [by_commit[hint] for hint in filename_hints if hint in by_commit]
        if len(hinted_indexes) == 1 and row_results[hinted_indexes[0]]["match_strength"] == "none":
            target_index = hinted_indexes[0]
            note_bits.append("Matched a non-canonical archived log using a unique short commit hint in the filename.")

        if target_index is None:
            parsed_val = parsed_log.get("summary_fields", {}).get("val_bpb")
            if parsed_val is not None:
                row_indexes = unmatched_by_val.get(f"{parsed_val:.6f}", [])
                if len(row_indexes) == 1:
                    target_index = row_indexes[0]
                    note_bits.append("Matched a non-canonical archived log using a unique val_bpb match.")

        if target_index is None or log_path in used_paths:
            continue

        candidate = {
            **row_results[target_index],
            "matched_log_path": str(log_path.resolve()),
            "match_type": "reconstructed with low confidence",
            "match_strength": "weak",
            "confidence": 0.4,
            "provenance_notes": " ".join(note_bits)
            + " This evidence is best-effort reconstruction rather than the canonical archive path.",
            "log_summary": log_summary_fields(parsed_log),
        }
        updated = consider_candidate(row_results[target_index], candidate)
        if updated is not row_results[target_index]:
            matched_val = row_results[target_index]["val_bpb"]
            row_results[target_index] = updated
            used_paths.add(log_path)
            unmatched_by_val[matched_val] = []

    return row_results


def build_output(
    results_path: Path,
    logs_dir: Path,
    latest_log_path: Path,
    rows: list[dict[str, str]],
    current_commit: str | None,
    current_commit_source: str,
) -> dict[str, Any]:
    latest_log = parse_log_file(latest_log_path)
    row_results = [make_base_row(row_index, row) for row_index, row in enumerate(rows)]

    for idx, row in enumerate(rows):
        row_results[idx] = archive_candidate(row, row_results[idx], logs_dir)
        row_results[idx] = latest_log_candidate(
            row,
            row_results[idx],
            latest_log_path,
            latest_log,
            current_commit,
            current_commit_source,
        )

    row_results = weak_reconstruction(rows, row_results, logs_dir)

    counts = Counter(result["match_type"] for result in row_results)
    summary_counts = {match_type: counts.get(match_type, 0) for match_type in MATCH_TYPE_COUNTS}
    return {
        "results_path": str(results_path.resolve()),
        "logs_dir": str(logs_dir.resolve()),
        "latest_log_path": str(latest_log_path.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": row_results,
        "summary": {
            "total_rows": len(row_results),
            "match_type_counts": summary_counts,
        },
    }


def main() -> int:
    args = parse_args()
    results_path = args.results.resolve()
    logs_dir = args.logs_dir.resolve()
    latest_log_path = args.latest_log.resolve()
    rows = read_results_rows(results_path)
    current_commit, current_commit_source = resolve_current_commit(results_path, args.current_commit)
    payload = build_output(results_path, logs_dir, latest_log_path, rows, current_commit, current_commit_source)
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
