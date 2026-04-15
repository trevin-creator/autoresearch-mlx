#!/usr/bin/env python3
"""Run one experiment, keep run.log, and archive durable per-run evidence."""

from __future__ import annotations

import argparse
import filecmp
import subprocess
import sys
from pathlib import Path


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_short_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def archive_run_log(run_log_path: Path, archive_path: Path) -> tuple[bool, str]:
    """Archive run.log to the durable location without clobbering mismatched evidence."""
    if not run_log_path.exists() or run_log_path.stat().st_size == 0:
        return True, f"No archival performed because {run_log_path} does not exist or is empty."

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        if filecmp.cmp(run_log_path, archive_path, shallow=False):
            return True, f"Archive already exists with identical bytes at {archive_path}."
        return (
            False,
            "Archive collision detected at "
            f"{archive_path}. The existing archive differs from {run_log_path}. "
            "This usually means the experiment was rerun without creating a new commit first.",
        )

    archive_path.write_bytes(run_log_path.read_bytes())
    return True, f"Archived run log to {archive_path}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--run-log", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_log_path = args.run_log.resolve() if args.run_log else repo_root / "run.log"
    logs_dir = args.logs_dir.resolve() if args.logs_dir else repo_root / "logs"

    short_commit = resolve_short_commit(repo_root)
    archive_path = logs_dir / f"{short_commit}.log"

    print(f"Running experiment for commit {short_commit}")
    print(f"Transient log: {run_log_path}")
    print(f"Archive path:  {archive_path}")

    with run_log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            ["uv", "run", "train.py"],
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    archived_ok, archive_message = archive_run_log(run_log_path, archive_path)
    print(archive_message, file=sys.stderr if not archived_ok else sys.stdout)

    if not archived_ok:
        return 2
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
