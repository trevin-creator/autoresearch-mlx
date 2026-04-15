#!/usr/bin/env python3
"""Shared helpers for parsing MLX experiment logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SUMMARY_FIELD_TYPES: dict[str, type] = {
    "val_bpb": float,
    "training_seconds": float,
    "total_seconds": float,
    "peak_vram_mb": float,
    "mfu_percent": float,
    "total_tokens_M": float,
    "num_steps": int,
    "num_params_M": float,
    "depth": int,
}
REQUIRED_FINAL_FIELDS = ("val_bpb", "peak_vram_mb", "training_seconds", "total_seconds")
SUMMARY_LINE_RE = re.compile(r"^(?P<key>[a-z_]+):\s+(?P<value>.+?)\s*$")
SHORT_COMMIT_RE = re.compile(r"\b([0-9a-f]{7})\b")


def extract_short_commit_hints(text: str) -> list[str]:
    """Return unique 7-char commit hints found in text."""
    return sorted(set(SHORT_COMMIT_RE.findall(text)))


def parse_log_text(text: str, *, source: str = "<memory>") -> dict[str, Any]:
    """Parse the final MLX summary block from a log."""
    summary_fields: dict[str, Any] = {}
    saw_summary_marker = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "---":
            saw_summary_marker = True
            continue

        match = SUMMARY_LINE_RE.match(line)
        if not match:
            continue

        key = match.group("key")
        if key not in SUMMARY_FIELD_TYPES:
            continue

        value_text = match.group("value")
        try:
            summary_fields[key] = SUMMARY_FIELD_TYPES[key](value_text)
        except ValueError:
            continue

    is_final_summary = all(field in summary_fields for field in REQUIRED_FINAL_FIELDS)
    return {
        "source": source,
        "summary_fields": summary_fields,
        "is_final_summary": is_final_summary,
        "saw_summary_marker": saw_summary_marker,
        "short_commit_hints": extract_short_commit_hints(text),
        "line_count": len(text.splitlines()),
    }


def parse_log_file(path: Path) -> dict[str, Any]:
    """Parse a log file without raising on missing files."""
    path = Path(path)
    if not path.exists():
        return {
            "source": str(path),
            "summary_fields": {},
            "is_final_summary": False,
            "saw_summary_marker": False,
            "short_commit_hints": [],
            "line_count": 0,
            "exists": False,
            "size_bytes": 0,
        }

    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log_text(text, source=str(path))
    parsed["exists"] = True
    parsed["size_bytes"] = path.stat().st_size
    return parsed
