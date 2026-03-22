"""Compact summary of best fixed/ternary/verilator candidates from JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize quantization search JSONLs")
    p.add_argument(
        "--fixed-jsonl",
        type=Path,
        default=Path("experiments/snn_fixed_trials.jsonl"),
    )
    p.add_argument(
        "--ternary-jsonl",
        type=Path,
        default=Path("experiments/snn_ternary_trials.jsonl"),
    )
    p.add_argument(
        "--verilator-jsonl",
        type=Path,
        default=Path("experiments/snn_ternary_verilator_trials.jsonl"),
    )
    p.add_argument(
        "--require-verilator-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, Verilator best candidate must have passed=True.",
    )
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _extract_candidate(
    row: dict[str, Any],
    *,
    require_verilator_pass: bool = False,
) -> dict[str, Any] | None:
    cfg = row.get("config")
    metrics = row.get("metrics")
    if not isinstance(cfg, dict) or not isinstance(metrics, dict):
        return None

    try:
        val_acc = float(metrics.get("val_acc", float("-inf")))
    except (TypeError, ValueError):
        return None

    mode = str(cfg.get("weight_mode", "unknown"))
    trial = row.get("trial_number", "-")
    study = row.get("study_name", "-")

    verilator = metrics.get("verilator")
    verilator_pass = None
    if isinstance(verilator, dict):
        verilator_pass = bool(verilator.get("passed", False))

    if require_verilator_pass and verilator_pass is not True:
        return None

    return {
        "study": str(study),
        "trial": str(trial),
        "val_acc": val_acc,
        "weight_mode": mode,
        "n_hidden": cfg.get("n_hidden", "-"),
        "n_layers": cfg.get("n_layers", "-"),
        "batch_size": cfg.get("batch_size", "-"),
        "learning_rate": cfg.get("learning_rate", "-"),
        "fixed_bits": cfg.get("fixed_point_bits", "-"),
        "fixed_frac": cfg.get("fixed_point_frac_bits", "-"),
        "fixed_round": cfg.get("fixed_point_round_mode", "-"),
        "ternary_threshold": cfg.get("ternary_threshold", "-"),
        "ternary_scale": cfg.get("ternary_scale_mode", "-"),
        "verilator_passed": verilator_pass,
    }


def _best_candidate(
    rows: list[dict[str, Any]],
    *,
    require_verilator_pass: bool = False,
) -> dict[str, Any] | None:
    candidates = [
        c
        for c in (
            _extract_candidate(r, require_verilator_pass=require_verilator_pass)
            for r in rows
        )
        if c is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: float(c["val_acc"]))


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "-"
    return str(value)


def _row(name: str, candidate: dict[str, Any] | None) -> list[str]:
    if candidate is None:
        return [name, "(no rows)", "-", "-", "-", "-", "-", "-", "-", "-"]
    return [
        name,
        _cell(candidate["study"]),
        _cell(candidate["trial"]),
        _cell(candidate["val_acc"]),
        _cell(candidate["weight_mode"]),
        _cell(candidate["n_hidden"]),
        _cell(candidate["n_layers"]),
        _cell(candidate["fixed_bits"]),
        _cell(candidate["ternary_threshold"]),
        _cell(candidate["verilator_passed"]),
    ]


def _print_table(rows: list[list[str]]) -> None:
    headers = [
        "category",
        "study",
        "trial",
        "val_acc",
        "mode",
        "n_hidden",
        "n_layers",
        "fixed_bits",
        "ternary_thr",
        "verilator_ok",
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main() -> None:
    args = parse_args()

    fixed_rows = _load_jsonl(args.fixed_jsonl)
    ternary_rows = _load_jsonl(args.ternary_jsonl)
    verilator_rows = _load_jsonl(args.verilator_jsonl)

    best_fixed = _best_candidate(fixed_rows)
    best_ternary = _best_candidate(ternary_rows)
    best_verilator = _best_candidate(
        verilator_rows,
        require_verilator_pass=args.require_verilator_pass,
    )

    table = [
        _row("fixed", best_fixed),
        _row("ternary", best_ternary),
        _row("verilator", best_verilator),
    ]

    _print_table(table)


if __name__ == "__main__":
    main()
