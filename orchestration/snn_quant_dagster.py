"""Dagster orchestration for fixed-point -> ternary -> Verilator SNN search."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dagster import Definitions, Failure, In, Permissive, job, op


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def run_stage_command(
    *,
    stage_name: str,
    command: list[str],
    timeout_s: int,
    log_jsonl: Path,
) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    result = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        timeout=max(1, timeout_s),
        check=False,
    )
    finished = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": 1,
        "timestamp_start_utc": started,
        "timestamp_end_utc": finished,
        "stage": stage_name,
        "command": command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
    append_jsonl(log_jsonl, payload)
    return payload


def run_fixed_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        "python",
        "search_snn_optuna.py",
        "--trials",
        str(cfg["trials"]),
        "--study-name",
        str(cfg["study_name"]),
        "--storage",
        str(cfg["storage"]),
        "--time-budget-s",
        str(cfg["time_budget_s"]),
        "--log-jsonl",
        str(cfg["log_jsonl"]),
        "--fixed-point-search",
        "--fixed-point-bits-min",
        str(cfg["fixed_point_bits_min"]),
        "--fixed-point-bits-max",
        str(cfg["fixed_point_bits_max"]),
        "--fixed-point-frac-min",
        str(cfg["fixed_point_frac_min"]),
        "--fixed-point-frac-max",
        str(cfg["fixed_point_frac_max"]),
        "--fixed-point-rounding-options",
        str(cfg["fixed_point_rounding_options"]),
    ]
    return run_stage_command(
        stage_name="fixed_search",
        command=cmd,
        timeout_s=int(cfg["timeout_s"]),
        log_jsonl=Path(cfg["stage_log_jsonl"]),
    )


def run_ternary_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        "python",
        "search_snn_optuna.py",
        "--trials",
        str(cfg["trials"]),
        "--study-name",
        str(cfg["study_name"]),
        "--storage",
        str(cfg["storage"]),
        "--time-budget-s",
        str(cfg["time_budget_s"]),
        "--log-jsonl",
        str(cfg["log_jsonl"]),
        "--ternary-search",
        "--ternary-threshold-min",
        str(cfg["ternary_threshold_min"]),
        "--ternary-threshold-max",
        str(cfg["ternary_threshold_max"]),
    ]
    return run_stage_command(
        stage_name="ternary_search",
        command=cmd,
        timeout_s=int(cfg["timeout_s"]),
        log_jsonl=Path(cfg["stage_log_jsonl"]),
    )


def run_verilator_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        "python",
        "search_snn_optuna.py",
        "--trials",
        str(cfg["trials"]),
        "--study-name",
        str(cfg["study_name"]),
        "--storage",
        str(cfg["storage"]),
        "--time-budget-s",
        str(cfg["time_budget_s"]),
        "--log-jsonl",
        str(cfg["log_jsonl"]),
        "--ternary-search",
        "--run-verilator",
        "--verilator-mode",
        str(cfg["verilator_mode"]),
        "--verilator-timeout-s",
        str(cfg["verilator_timeout_s"]),
        "--verilator-top",
        str(cfg["verilator_top"]),
        "--verilator-max-width",
        str(cfg["verilator_max_width"]),
        "--verilator-sim-width",
        str(cfg["verilator_sim_width"]),
        "--verilator-sim-steps",
        str(cfg["verilator_sim_steps"]),
    ]
    return run_stage_command(
        stage_name="ternary_verilator",
        command=cmd,
        timeout_s=int(cfg["timeout_s"]),
        log_jsonl=Path(cfg["stage_log_jsonl"]),
    )


@op(config_schema={"fixed": Permissive()})
def fixed_search_op(context) -> dict[str, Any]:
    result = run_fixed_stage(context.op_config["fixed"])
    if not result["passed"]:
        raise Failure(description="fixed_search failed")
    return result


@op(ins={"fixed_result": In(dict)}, config_schema={"ternary": Permissive()})
def ternary_search_op(
    context, fixed_result: dict[str, Any]
) -> dict[str, Any]:
    if not fixed_result.get("passed", False):
        raise Failure(description="fixed_search did not pass")
    result = run_ternary_stage(context.op_config["ternary"])
    if not result["passed"]:
        raise Failure(description="ternary_search failed")
    return result


@op(ins={"ternary_result": In(dict)}, config_schema={"verilator": Permissive()})
def verilator_op(
    context, ternary_result: dict[str, Any]
) -> dict[str, Any]:
    if not ternary_result.get("passed", False):
        raise Failure(description="ternary_search did not pass")
    result = run_verilator_stage(context.op_config["verilator"])
    if not result["passed"]:
        raise Failure(description="ternary_verilator failed")
    return result


@job
def snn_quant_pipeline() -> None:
    verilator_op(ternary_search_op(fixed_search_op()))


defs = Definitions(jobs=[snn_quant_pipeline])
