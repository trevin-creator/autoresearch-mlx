"""Run fixed-point -> ternary -> Verilator SNN search with Dagster-style staging."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from orchestration.snn_quant_dagster import (
    append_jsonl,
    run_fixed_stage,
    run_ternary_stage,
    run_verilator_stage,
    snn_quant_pipeline,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run staged SNN quantization pipeline")
    p.add_argument("--params", type=Path, default=Path("params_snn_quant.yaml"))
    p.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["fixed", "ternary", "verilator", "full"],
        help="Stage target. full/verilator runs fixed->ternary->verilator.",
    )
    p.add_argument(
        "--skip-prereqs",
        action="store_true",
        help="Run only the selected stage (DVC stages use this).",
    )
    p.add_argument(
        "--pipeline-log-jsonl",
        type=Path,
        default=Path("experiments/quant_pipeline/pipeline_runs.jsonl"),
    )
    p.add_argument(
        "--use-dagster",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Dagster job execution for full-stage orchestration.",
    )
    return p.parse_args()


def load_params(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("Top-level params file must be a mapping")
    return data


def _mark_pipeline_event(
    *,
    path: Path,
    stage: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "status": status,
        "details": details or {},
    }
    append_jsonl(path, payload)


def main() -> None:
    args = parse_args()
    params = load_params(args.params)

    fixed_cfg = dict(params.get("fixed_search", {}))
    ternary_cfg = dict(params.get("ternary_search", {}))
    verilator_cfg = dict(params.get("verilator_stage", {}))

    run_target = "verilator" if args.stage == "full" else args.stage

    _mark_pipeline_event(
        path=args.pipeline_log_jsonl,
        stage=run_target,
        status="started",
        details={
            "params": str(args.params),
            "skip_prereqs": args.skip_prereqs,
        },
    )

    fixed_result: dict[str, Any] = {"passed": True}
    ternary_result: dict[str, Any] = {"passed": True}

    if run_target == "verilator" and not args.skip_prereqs and args.use_dagster:
        run_config = {
            "ops": {
                "fixed_search_op": {"config": {"fixed": fixed_cfg}},
                "ternary_search_op": {"config": {"ternary": ternary_cfg}},
                "verilator_op": {"config": {"verilator": verilator_cfg}},
            }
        }
        dagster_result = snn_quant_pipeline.execute_in_process(run_config=run_config)
        if not dagster_result.success:
            _mark_pipeline_event(
                path=args.pipeline_log_jsonl,
                stage="dagster_full",
                status="failed",
                details={"reason": "Dagster pipeline failed"},
            )
            raise SystemExit(1)

        _mark_pipeline_event(
            path=args.pipeline_log_jsonl,
            stage="dagster_full",
            status="completed",
            details={"params": str(args.params)},
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "stage": run_target,
                    "mode": "dagster",
                    "pipeline_log": str(args.pipeline_log_jsonl),
                },
                indent=2,
            )
        )
        return

    if run_target in {"fixed", "ternary", "verilator"} and not args.skip_prereqs:
        fixed_result = run_fixed_stage(fixed_cfg)
        if not fixed_result["passed"]:
            _mark_pipeline_event(
                path=args.pipeline_log_jsonl,
                stage="fixed",
                status="failed",
                details=fixed_result,
            )
            raise SystemExit(1)

    if run_target in {"ternary", "verilator"}:
        if args.skip_prereqs:
            pass
        elif not fixed_result.get("passed", False):
            _mark_pipeline_event(
                path=args.pipeline_log_jsonl,
                stage="ternary",
                status="skipped",
                details={"reason": "fixed stage failed"},
            )
            raise SystemExit(1)

        if run_target == "ternary" or not args.skip_prereqs:
            ternary_result = run_ternary_stage(ternary_cfg)
            if not ternary_result["passed"]:
                _mark_pipeline_event(
                    path=args.pipeline_log_jsonl,
                    stage="ternary",
                    status="failed",
                    details=ternary_result,
                )
                raise SystemExit(1)

    if run_target == "verilator":
        if not args.skip_prereqs and not ternary_result.get("passed", False):
            _mark_pipeline_event(
                path=args.pipeline_log_jsonl,
                stage="verilator",
                status="skipped",
                details={"reason": "ternary stage failed"},
            )
            raise SystemExit(1)

        verilator_result = run_verilator_stage(verilator_cfg)
        if not verilator_result["passed"]:
            _mark_pipeline_event(
                path=args.pipeline_log_jsonl,
                stage="verilator",
                status="failed",
                details=verilator_result,
            )
            raise SystemExit(1)

    _mark_pipeline_event(
        path=args.pipeline_log_jsonl,
        stage=run_target,
        status="completed",
        details={"params": str(args.params)},
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "stage": run_target,
                "pipeline_log": str(args.pipeline_log_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
