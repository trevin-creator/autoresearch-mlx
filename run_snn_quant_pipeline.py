"""Run quant search and full autoresearch hardware pipeline with Dagster."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from orchestration.snn_quant_dagster import (
    append_jsonl,
    run_data_generation_stage,
    run_fixed_reference_stage,
    run_fixed_stage,
    run_fpga_synthesis_stage,
    run_icarus_stage,
    run_rtl_stage,
    run_snn_ir_stage,
    run_spyx_float_stage,
    run_ternary_stage,
    run_verilator_integration_stage,
    run_verilator_stage,
    snn_autoresearch_hw_pipeline,
    snn_quant_pipeline,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run staged SNN quantization pipeline")
    p.add_argument("--params", type=Path, default=Path("params_snn_quant.yaml"))
    p.add_argument(
        "--profile",
        type=str,
        default="full",
        help="Parameter profile name from params file (e.g. full, smoke).",
    )
    p.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=[
            "fixed",
            "ternary",
            "verilator",
            "data_generation",
            "spyx_float",
            "fixed_reference",
            "snn_ir",
            "rtl",
            "icarus",
            "verilator_integration",
            "fpga_synthesis",
            "full",
        ],
        help=(
            "Stage target. full runs the complete flow: "
            "dataset->spyx(float)->fixed_ref->ternary->snn_ir->rtl->icarus->verilator->fpga"
        ),
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


def select_profile_params(params: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = params.get("profiles")
    if not isinstance(profiles, dict):
        return params

    selected = profiles.get(profile)
    if not isinstance(selected, dict):
        available = ", ".join(sorted(str(k) for k in profiles))
        msg = f"Unknown profile '{profile}'. Available: {available}"
        raise KeyError(msg)
    return selected


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


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    params = select_profile_params(load_params(args.params), args.profile)

    fixed_cfg = dict(params.get("fixed_search", {}))
    ternary_cfg = dict(params.get("ternary_search", {}))
    verilator_cfg = dict(params.get("verilator_stage", {}))
    data_cfg = dict(params.get("data_generation_stage", {}))
    spyx_cfg = dict(params.get("spyx_float_stage", {}))
    fixed_ref_cfg = dict(params.get("fixed_point_reference_stage", {}))
    snn_ir_cfg = dict(params.get("snn_ir_stage", {}))
    rtl_cfg = dict(params.get("rtl_stage", {}))
    icarus_cfg = dict(params.get("icarus_stage", {}))
    verilator_integration_cfg = dict(params.get("verilator_integration_stage", {}))
    fpga_cfg = dict(params.get("fpga_synthesis_stage", {}))

    run_target = "fpga_synthesis" if args.stage == "full" else args.stage

    _mark_pipeline_event(
        path=args.pipeline_log_jsonl,
        stage=run_target,
        status="started",
        details={
            "params": str(args.params),
            "profile": args.profile,
            "skip_prereqs": args.skip_prereqs,
        },
    )

    fixed_result: dict[str, Any] = {"passed": True}
    ternary_result: dict[str, Any] = {"passed": True}

    quant_only_mode = not data_cfg

    if run_target == "fpga_synthesis" and not args.skip_prereqs and args.use_dagster:
        if quant_only_mode:
            run_config = {
                "ops": {
                    "fixed_search_op": {"config": {"fixed": fixed_cfg}},
                    "ternary_search_op": {"config": {"ternary": ternary_cfg}},
                    "verilator_op": {"config": {"verilator": verilator_cfg}},
                }
            }
            dagster_result = snn_quant_pipeline.execute_in_process(
                run_config=run_config
            )
        else:
            run_config = {
                "ops": {
                    "data_generation_op": {"config": {"data_generation": data_cfg}},
                    "spyx_float_op": {"config": {"spyx_float": spyx_cfg}},
                    "fixed_ref_op": {"config": {"fixed_ref": fixed_ref_cfg}},
                    "ternary_hw_op": {"config": {"ternary": ternary_cfg}},
                    "snn_ir_op": {"config": {"snn_ir": snn_ir_cfg}},
                    "rtl_codegen_op": {"config": {"rtl": rtl_cfg}},
                    "icarus_op": {"config": {"icarus": icarus_cfg}},
                    "verilator_integration_op": {
                        "config": {"verilator_integration": verilator_integration_cfg}
                    },
                    "fpga_synth_op": {"config": {"fpga_synth": fpga_cfg}},
                }
            }
            dagster_result = snn_autoresearch_hw_pipeline.execute_in_process(
                run_config=run_config
            )

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
                    "profile": args.profile,
                    "pipeline_log": str(args.pipeline_log_jsonl),
                },
                indent=2,
            )
        )
        return

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
                    "profile": args.profile,
                    "pipeline_log": str(args.pipeline_log_jsonl),
                },
                indent=2,
            )
        )
        return

    if run_target == "data_generation":
        result = run_data_generation_stage(data_cfg)
        if not result["passed"]:
            raise SystemExit(1)

    if run_target == "spyx_float":
        result = run_spyx_float_stage(spyx_cfg)
        if not result["passed"]:
            raise SystemExit(1)

    if run_target == "fixed_reference":
        result = run_fixed_reference_stage(fixed_ref_cfg)
        if not result["passed"]:
            raise SystemExit(1)

    if run_target == "snn_ir":
        result = run_snn_ir_stage(snn_ir_cfg)
        if not result["passed"]:
            raise SystemExit(1)

    if run_target == "rtl":
        result = run_rtl_stage(rtl_cfg)
        if not result["passed"]:
            raise SystemExit(1)

    if run_target == "icarus":
        rtl_result = run_rtl_stage(rtl_cfg)
        if not rtl_result["passed"]:
            raise SystemExit(1)
        result = run_icarus_stage(icarus_cfg, rtl_result)
        if not result["passed"]:
            raise SystemExit(1)

    if run_target == "verilator_integration":
        rtl_result = run_rtl_stage(rtl_cfg)
        if not rtl_result["passed"]:
            raise SystemExit(1)
        result = run_verilator_integration_stage(verilator_integration_cfg, rtl_result)
        if not result["passed"]:
            raise SystemExit(1)

    if run_target == "fpga_synthesis" and args.skip_prereqs:
        result = run_fpga_synthesis_stage(fpga_cfg)
        if not result["passed"]:
            raise SystemExit(1)

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
                "profile": args.profile,
                "pipeline_log": str(args.pipeline_log_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
