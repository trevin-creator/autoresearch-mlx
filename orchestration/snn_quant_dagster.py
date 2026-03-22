"""Dagster orchestration for quant search and full autoresearch hardware flow."""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dagster import Definitions, Failure, In, Permissive, job, op


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def run_stage_command(
    *,
    stage_name: str,
    command: list[str],
    timeout_s: int,
    log_jsonl: Path,
    cwd: str | None = None,
) -> dict[str, Any]:
    started = _now_utc()
    result = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        timeout=max(1, timeout_s),
        check=False,
        cwd=cwd,
    )
    finished = _now_utc()
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
        "cwd": cwd or ".",
    }
    append_jsonl(log_jsonl, payload)
    return payload


def _stage_log_path(cfg: dict[str, Any]) -> Path:
    return Path(
        str(cfg.get("stage_log_jsonl", "experiments/quant_pipeline/stage_runs.jsonl"))
    )


def _run_config_command(stage_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    command_text = str(cfg.get("command", "")).strip()
    if not command_text:
        msg = f"{stage_name}: missing 'command' in config"
        raise ValueError(msg)

    return run_stage_command(
        stage_name=stage_name,
        command=shlex.split(command_text),
        timeout_s=int(cfg.get("timeout_s", 600)),
        log_jsonl=_stage_log_path(cfg),
        cwd=str(cfg.get("cwd", ".")),
    )


def run_data_generation_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_config_command("data_generation", cfg)


def run_spyx_float_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_config_command("spyx_float_mlx", cfg)


def run_fixed_reference_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_config_command("fixed_point_reference", cfg)


def run_ternary_search_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_config_command("ternary_search", cfg)


def run_snn_ir_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_config_command("snn_ir_export", cfg)


def run_rtl_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    ir_json = Path(str(cfg["ir_json"]))
    rtl_sv = Path(str(cfg["rtl_sv"]))
    tb_sv = Path(str(cfg.get("tb_sv", rtl_sv.with_name("tb_top.sv"))))
    stage_log = Path(
        str(cfg.get("stage_log_jsonl", "experiments/quant_pipeline/stage_runs.jsonl"))
    )

    started = _now_utc()

    from snn_ir.codegen.sv import generate_sv  # noqa: PLC0415
    from snn_ir.schema import NetworkIR  # noqa: PLC0415

    ir = NetworkIR.model_validate_json(ir_json.read_text(encoding="utf-8"))
    rtl_sv.parent.mkdir(parents=True, exist_ok=True)
    tb_sv.parent.mkdir(parents=True, exist_ok=True)

    rtl_text = generate_sv(ir)
    rtl_sv.write_text(rtl_text, encoding="utf-8")

    top_module = f"{ir.name}_top"
    tb_top = str(cfg.get("tb_top", "tb_top"))

    tb_text = f"""module {tb_top};
  logic clk = 1'b0;
  logic rst = 1'b1;

  always #5 clk = ~clk;

  initial begin
    repeat (2) @(posedge clk);
    rst = 1'b0;
    repeat (8) @(posedge clk);
    $display(\"TB_DONE\");
    $finish;
  end

  {top_module} dut (
    .clk(clk),
    .rst(rst),
    .in_bits('0),
    .out_spk()
  );
endmodule
"""
    tb_sv.write_text(tb_text, encoding="utf-8")

    payload = {
        "schema_version": 1,
        "timestamp_start_utc": started,
        "timestamp_end_utc": _now_utc(),
        "stage": "rtl_codegen",
        "passed": True,
        "ir_json": str(ir_json),
        "rtl_sv": str(rtl_sv),
        "tb_sv": str(tb_sv),
        "top_module": top_module,
        "tb_top": tb_top,
    }
    append_jsonl(stage_log, payload)
    return payload


def run_icarus_stage(cfg: dict[str, Any], rtl_result: dict[str, Any]) -> dict[str, Any]:
    rtl_sv = str(rtl_result["rtl_sv"])
    tb_sv = str(rtl_result["tb_sv"])
    tb_top = str(rtl_result["tb_top"])

    sim_out = Path(str(cfg.get("sim_out", "experiments/hw_flow/icarus_sim.out")))
    sim_out.parent.mkdir(parents=True, exist_ok=True)

    compile_cmd = [
        "iverilog",
        "-g2012",
        "-s",
        tb_top,
        "-o",
        str(sim_out),
        rtl_sv,
        tb_sv,
    ]

    compile_result = run_stage_command(
        stage_name="icarus_compile",
        command=compile_cmd,
        timeout_s=int(cfg.get("timeout_s", 600)),
        log_jsonl=_stage_log_path(cfg),
        cwd=str(cfg.get("cwd", ".")),
    )
    if not compile_result["passed"]:
        return compile_result

    run_result = run_stage_command(
        stage_name="icarus_functional_sim",
        command=["vvp", str(sim_out)],
        timeout_s=int(cfg.get("timeout_s", 600)),
        log_jsonl=_stage_log_path(cfg),
        cwd=str(cfg.get("cwd", ".")),
    )

    run_result["rtl_sv"] = rtl_sv
    run_result["tb_sv"] = tb_sv
    run_result["tb_top"] = tb_top
    run_result["sim_out"] = str(sim_out)
    return run_result


def run_verilator_integration_stage(
    cfg: dict[str, Any], rtl_result: dict[str, Any]
) -> dict[str, Any]:
    rtl_sv = str(rtl_result["rtl_sv"])
    tb_sv = str(rtl_result["tb_sv"])
    tb_top = str(rtl_result["tb_top"])

    mode = str(cfg.get("mode", "lint"))
    if mode == "lint":
        cmd = [
            "verilator",
            "--lint-only",
            "-Wall",
            "--top-module",
            tb_top,
            rtl_sv,
            tb_sv,
        ]
    else:
        cmd_text = str(cfg.get("command", "")).strip()
        if not cmd_text:
            raise ValueError("verilator_integration: provide command when mode != lint")
        cmd = shlex.split(cmd_text)

    result = run_stage_command(
        stage_name="verilator_integration",
        command=cmd,
        timeout_s=int(cfg.get("timeout_s", 600)),
        log_jsonl=_stage_log_path(cfg),
        cwd=str(cfg.get("cwd", ".")),
    )
    result["rtl_sv"] = rtl_sv
    result["tb_sv"] = tb_sv
    result["tb_top"] = tb_top
    return result


def run_fpga_synthesis_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_config_command("fpga_synthesis", cfg)


# Legacy quant stages (kept for compatibility)
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
        log_jsonl=Path(str(cfg["stage_log_jsonl"])),
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
        log_jsonl=Path(str(cfg["stage_log_jsonl"])),
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
        log_jsonl=Path(str(cfg["stage_log_jsonl"])),
    )


@op(config_schema={"fixed": Permissive()})
def fixed_search_op(context) -> dict[str, Any]:
    result = run_fixed_stage(context.op_config["fixed"])
    if not result["passed"]:
        raise Failure(description="fixed_search failed")
    return result


@op(ins={"fixed_result": In(dict)}, config_schema={"ternary": Permissive()})
def ternary_search_op(context, fixed_result: dict[str, Any]) -> dict[str, Any]:
    if not fixed_result.get("passed", False):
        raise Failure(description="fixed_search did not pass")
    result = run_ternary_stage(context.op_config["ternary"])
    if not result["passed"]:
        raise Failure(description="ternary_search failed")
    return result


@op(ins={"ternary_result": In(dict)}, config_schema={"verilator": Permissive()})
def verilator_op(context, ternary_result: dict[str, Any]) -> dict[str, Any]:
    if not ternary_result.get("passed", False):
        raise Failure(description="ternary_search did not pass")
    result = run_verilator_stage(context.op_config["verilator"])
    if not result["passed"]:
        raise Failure(description="ternary_verilator failed")
    return result


@job
def snn_quant_pipeline() -> None:
    verilator_op(ternary_search_op(fixed_search_op()))


@op(config_schema={"data_generation": Permissive()})
def data_generation_op(context) -> dict[str, Any]:
    result = run_data_generation_stage(context.op_config["data_generation"])
    if not result["passed"]:
        raise Failure(description="data_generation failed")
    return result


@op(ins={"data_result": In(dict)}, config_schema={"spyx_float": Permissive()})
def spyx_float_op(context, data_result: dict[str, Any]) -> dict[str, Any]:
    if not data_result.get("passed", False):
        raise Failure(description="data_generation did not pass")
    result = run_spyx_float_stage(context.op_config["spyx_float"])
    if not result["passed"]:
        raise Failure(description="spyx_float_mlx failed")
    return result


@op(ins={"spyx_result": In(dict)}, config_schema={"fixed_ref": Permissive()})
def fixed_ref_op(context, spyx_result: dict[str, Any]) -> dict[str, Any]:
    if not spyx_result.get("passed", False):
        raise Failure(description="spyx_float_mlx did not pass")
    result = run_fixed_reference_stage(context.op_config["fixed_ref"])
    if not result["passed"]:
        raise Failure(description="fixed_point_reference failed")
    return result


@op(ins={"fixed_ref_result": In(dict)}, config_schema={"ternary": Permissive()})
def ternary_hw_op(context, fixed_ref_result: dict[str, Any]) -> dict[str, Any]:
    if not fixed_ref_result.get("passed", False):
        raise Failure(description="fixed_point_reference did not pass")
    result = run_ternary_search_stage(context.op_config["ternary"])
    if not result["passed"]:
        raise Failure(description="ternary_search failed")
    return result


@op(ins={"ternary_result": In(dict)}, config_schema={"snn_ir": Permissive()})
def snn_ir_op(context, ternary_result: dict[str, Any]) -> dict[str, Any]:
    if not ternary_result.get("passed", False):
        raise Failure(description="ternary_search did not pass")
    result = run_snn_ir_stage(context.op_config["snn_ir"])
    if not result["passed"]:
        raise Failure(description="snn_ir_export failed")
    return result


@op(ins={"snn_ir_result": In(dict)}, config_schema={"rtl": Permissive()})
def rtl_codegen_op(context, snn_ir_result: dict[str, Any]) -> dict[str, Any]:
    if not snn_ir_result.get("passed", False):
        raise Failure(description="snn_ir_export did not pass")
    result = run_rtl_stage(context.op_config["rtl"])
    if not result["passed"]:
        raise Failure(description="rtl_codegen failed")
    return result


@op(ins={"rtl_result": In(dict)}, config_schema={"icarus": Permissive()})
def icarus_op(context, rtl_result: dict[str, Any]) -> dict[str, Any]:
    result = run_icarus_stage(context.op_config["icarus"], rtl_result)
    if not result["passed"]:
        raise Failure(description="icarus_functional_sim failed")
    return result


@op(
    ins={"icarus_result": In(dict), "rtl_result": In(dict)},
    config_schema={"verilator_integration": Permissive()},
)
def verilator_integration_op(
    context, icarus_result: dict[str, Any], rtl_result: dict[str, Any]
) -> dict[str, Any]:
    if not icarus_result.get("passed", False):
        raise Failure(description="icarus_functional_sim did not pass")
    result = run_verilator_integration_stage(
        context.op_config["verilator_integration"], rtl_result
    )
    if not result["passed"]:
        raise Failure(description="verilator_integration failed")
    return result


@op(
    ins={"verilator_result": In(dict)},
    config_schema={"fpga_synth": Permissive()},
)
def fpga_synth_op(context, verilator_result: dict[str, Any]) -> dict[str, Any]:
    if not verilator_result.get("passed", False):
        raise Failure(description="verilator_integration did not pass")
    result = run_fpga_synthesis_stage(context.op_config["fpga_synth"])
    if not result["passed"]:
        raise Failure(description="fpga_synthesis failed")
    return result


@job
def snn_autoresearch_hw_pipeline() -> None:
    data = data_generation_op()
    spyx = spyx_float_op(data)
    fixed = fixed_ref_op(spyx)
    ternary = ternary_hw_op(fixed)
    ir = snn_ir_op(ternary)
    rtl = rtl_codegen_op(ir)
    icarus = icarus_op(rtl)
    verilator = verilator_integration_op(icarus, rtl)
    fpga_synth_op(verilator)


defs = Definitions(jobs=[snn_quant_pipeline, snn_autoresearch_hw_pipeline])
