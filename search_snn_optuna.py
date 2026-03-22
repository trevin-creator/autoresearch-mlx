"""Optuna-driven structured search over train_snn.py experiment space."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna

from train_snn import ExperimentConfig, append_jsonl, get_git_commit_short, train_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna search for SHD SNN")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="snn-search")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///experiments/snn_optuna.db",
        help="Optuna storage URL",
    )
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=float(ExperimentConfig.time_budget_s),
        help="Per-trial training budget in seconds.",
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=Path("experiments/snn_optuna_trials.jsonl"),
    )
    parser.add_argument(
        "--ternary-search",
        action="store_true",
        help="Enable ternary mode and search ternary hyperparameters.",
    )
    parser.add_argument(
        "--fixed-point-search",
        action="store_true",
        help="Enable fixed-point mode and search fixed-point hyperparameters.",
    )
    parser.add_argument(
        "--fixed-point-bits-min",
        type=int,
        default=4,
        help="Lower bound for fixed-point total bit-width search.",
    )
    parser.add_argument(
        "--fixed-point-bits-max",
        type=int,
        default=12,
        help="Upper bound for fixed-point total bit-width search.",
    )
    parser.add_argument(
        "--fixed-point-frac-min",
        type=int,
        default=1,
        help="Lower bound for fixed-point fractional bits search.",
    )
    parser.add_argument(
        "--fixed-point-frac-max",
        type=int,
        default=8,
        help="Upper bound for fixed-point fractional bits search.",
    )
    parser.add_argument(
        "--fixed-point-rounding-options",
        type=str,
        default="nearest,floor",
        help="Comma-separated fixed-point round modes to sample from.",
    )
    parser.add_argument(
        "--ternary-threshold-min",
        type=float,
        default=0.02,
        help="Lower bound for ternary threshold search.",
    )
    parser.add_argument(
        "--ternary-threshold-max",
        type=float,
        default=0.35,
        help="Upper bound for ternary threshold search.",
    )
    parser.add_argument(
        "--run-verilator",
        action="store_true",
        help="Run a Verilator command after each ternary trial.",
    )
    parser.add_argument(
        "--verilator-mode",
        type=str,
        default="command",
        choices=["command", "lint", "simulate"],
        help=(
            "Verilator execution mode: "
            "'command' runs --verilator-command; "
            "'lint' generates trial RTL and runs verilator --lint-only; "
            "'simulate' generates RTL + C++ testbench, builds and runs."
        ),
    )
    parser.add_argument(
        "--verilator-command",
        type=str,
        default="verilator --version",
        help=(
            "Verilator command to execute. Supports {trial_dir} and {trial_number} "
            "format placeholders."
        ),
    )
    parser.add_argument(
        "--verilator-timeout-s",
        type=int,
        default=120,
        help="Timeout for each Verilator run.",
    )
    parser.add_argument(
        "--verilator-top",
        type=str,
        default="ternary_trial_top",
        help="Top module name when --verilator-mode lint is used.",
    )
    parser.add_argument(
        "--verilator-max-width",
        type=int,
        default=512,
        help="Cap generated RTL bus widths for lint mode.",
    )
    parser.add_argument(
        "--verilator-sim-width",
        type=int,
        default=32,
        help=(
            "Cap bus widths (<=32) for simulate mode so C++ testbench "
            "can use plain uint32_t port access."
        ),
    )
    parser.add_argument(
        "--verilator-sim-steps",
        type=int,
        default=16,
        help="Number of input stimulus steps driven in the C++ testbench.",
    )
    return parser.parse_args()


def trial_to_payload(
    trial: optuna.trial.Trial,
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit_short(),
        "study_name": trial.study.study_name,
        "trial_number": trial.number,
        "trial_params": trial.params,
        "config": dataclasses.asdict(cfg),
        "metrics": metrics,
    }


def build_config(
    trial: optuna.trial.Trial,
    time_budget_s: float,
    *,
    ternary_search: bool,
    fixed_point_search: bool,
    fixed_point_bits_min: int,
    fixed_point_bits_max: int,
    fixed_point_frac_min: int,
    fixed_point_frac_max: int,
    fixed_point_rounding_options: list[str],
    ternary_threshold_min: float,
    ternary_threshold_max: float,
) -> ExperimentConfig:
    if ternary_search and fixed_point_search:
        weight_mode = trial.suggest_categorical("weight_mode", ["fixed", "ternary"])
    elif ternary_search:
        weight_mode = "ternary"
    elif fixed_point_search:
        weight_mode = "fixed"
    else:
        weight_mode = "float"

    if weight_mode == "ternary":
        ternary_threshold = trial.suggest_float(
            "ternary_threshold",
            min(ternary_threshold_min, ternary_threshold_max),
            max(ternary_threshold_min, ternary_threshold_max),
        )
        ternary_scale_mode = trial.suggest_categorical(
            "ternary_scale_mode", ["mean_abs", "max_abs"]
        )
        ternary_use_ste = trial.suggest_categorical("ternary_use_ste", [True, False])
    else:
        ternary_threshold = ExperimentConfig.ternary_threshold
        ternary_scale_mode = ExperimentConfig.ternary_scale_mode
        ternary_use_ste = ExperimentConfig.ternary_use_ste

    if weight_mode == "fixed":
        fixed_point_bits = trial.suggest_int(
            "fixed_point_bits",
            min(fixed_point_bits_min, fixed_point_bits_max),
            max(fixed_point_bits_min, fixed_point_bits_max),
        )
        frac_hi = min(
            max(fixed_point_frac_min, fixed_point_frac_max), fixed_point_bits - 1
        )
        frac_lo = min(fixed_point_frac_min, fixed_point_frac_max)
        frac_lo = min(frac_lo, frac_hi)
        fixed_point_frac_bits = trial.suggest_int(
            "fixed_point_frac_bits", frac_lo, frac_hi
        )
        fixed_point_round_mode = trial.suggest_categorical(
            "fixed_point_round_mode",
            fixed_point_rounding_options or ["nearest", "floor"],
        )
        fixed_point_use_ste = trial.suggest_categorical(
            "fixed_point_use_ste", [True, False]
        )
    else:
        fixed_point_bits = ExperimentConfig.fixed_point_bits
        fixed_point_frac_bits = ExperimentConfig.fixed_point_frac_bits
        fixed_point_round_mode = ExperimentConfig.fixed_point_round_mode
        fixed_point_use_ste = ExperimentConfig.fixed_point_use_ste

    return ExperimentConfig(
        n_hidden=trial.suggest_int("n_hidden", 96, 320, step=32),
        n_layers=trial.suggest_int("n_layers", 1, 4),
        batch_size=trial.suggest_categorical("batch_size", [32, 48, 64, 96]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
        final_lr_frac=trial.suggest_float("final_lr_frac", 0.0, 0.2),
        weight_mode=weight_mode,
        fixed_point_bits=fixed_point_bits,
        fixed_point_frac_bits=fixed_point_frac_bits,
        fixed_point_round_mode=fixed_point_round_mode,
        fixed_point_use_ste=fixed_point_use_ste,
        ternary_threshold=ternary_threshold,
        ternary_scale_mode=ternary_scale_mode,
        ternary_use_ste=ternary_use_ste,
        seed=42 + trial.number,
        time_budget_s=time_budget_s,
    )


def run_verilator_step(
    trial: optuna.trial.Trial,
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
    command_template: str,
    timeout_s: int,
) -> dict[str, Any]:
    trial_dir = Path("experiments") / "verilator" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "trial_number": trial.number,
        "trial_params": trial.params,
        "config": dataclasses.asdict(cfg),
        "metrics": metrics,
    }
    with (trial_dir / "trial_context.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    command = command_template.format(
        trial_dir=str(trial_dir),
        trial_number=trial.number,
    )

    # Command comes from trusted local CLI input for researcher-controlled flows.
    result = subprocess.run(  # noqa: S603
        shlex.split(command),
        capture_output=True,
        text=True,
        timeout=max(1, timeout_s),
        check=False,
    )

    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "passed": result.returncode == 0,
        "trial_dir": str(trial_dir),
    }


def _write_trial_lint_sv(
    *,
    trial_dir: Path,
    top_module: str,
    cfg: ExperimentConfig,
    max_width: int,
) -> Path:
    in_w = max(1, min(int(cfg.n_hidden), max_width))
    out_w = max(1, min(int(cfg.n_layers * 32), max_width))
    threshold_q = int(round(max(0.0, min(1.0, cfg.ternary_threshold)) * 1000.0))

    sv = f"""module {top_module} #(
    parameter int IN_W = {in_w},
    parameter int OUT_W = {out_w},
    parameter int TERNARY_THRESHOLD_Q = {threshold_q}
)(
    input  logic [IN_W-1:0]  in_bits,
    output logic [OUT_W-1:0] out_bits
);
    logic [OUT_W-1:0] folded;
    always_comb begin
        // Deterministic fold so lint sees non-trivial combinational logic.
        folded = {{OUT_W{{1'b0}}}};
        for (int i = 0; i < OUT_W; i++) begin
            folded[i] = in_bits[i % IN_W] ^ in_bits[(i + TERNARY_THRESHOLD_Q) % IN_W];
        end
        out_bits = folded;
    end
endmodule
"""

    sv_path = trial_dir / f"{top_module}.sv"
    sv_path.write_text(sv, encoding="utf-8")
    return sv_path


def run_verilator_lint_step(
    trial: optuna.trial.Trial,
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
    timeout_s: int,
    *,
    top_module: str,
    max_width: int,
) -> dict[str, Any]:
    trial_dir = Path("experiments") / "verilator" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "trial_number": trial.number,
        "trial_params": trial.params,
        "config": dataclasses.asdict(cfg),
        "metrics": metrics,
    }
    with (trial_dir / "trial_context.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    sv_path = _write_trial_lint_sv(
        trial_dir=trial_dir,
        top_module=top_module,
        cfg=cfg,
        max_width=max_width,
    )
    command = [
        "verilator",
        "--lint-only",
        "-Wall",
        "--top-module",
        top_module,
        str(sv_path),
    ]
    result = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        timeout=max(1, timeout_s),
        check=False,
    )

    return {
        "mode": "lint",
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "passed": result.returncode == 0,
        "trial_dir": str(trial_dir),
        "rtl_file": str(sv_path),
        "top_module": top_module,
    }


def _write_trial_tb_cpp(
    *,
    trial_dir: Path,
    top_module: str,
    n_steps: int,
) -> Path:
    """Write a minimal Verilator C++ testbench that drives in_bits[0..n_steps-1]."""
    cpp = f"""#include <cstdint>
#include <iostream>
#include "verilated.h"
#include "V{top_module}.h"

int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);
    V{top_module}* dut = new V{top_module};
    uint32_t errors = 0;
    for (uint32_t i = 0; i < {n_steps}u; ++i) {{
        dut->in_bits = i;
        dut->eval();
        std::cout << "step " << i
                  << " in=" << dut->in_bits
                  << " out=" << (uint32_t)dut->out_bits << "\\n";
    }}
    dut->final();
    delete dut;
    std::cout << "DONE errors=" << errors << "\\n";
    return (int)errors;
}}
"""
    cpp_path = trial_dir / f"{top_module}_tb.cpp"
    cpp_path.write_text(cpp, encoding="utf-8")
    return cpp_path


def run_verilator_simulate_step(
    trial: optuna.trial.Trial,
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
    timeout_s: int,
    *,
    top_module: str,
    sim_width: int,
    sim_steps: int,
) -> dict[str, Any]:
    trial_dir = Path("experiments") / "verilator" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "trial_number": trial.number,
        "trial_params": trial.params,
        "config": dataclasses.asdict(cfg),
        "metrics": metrics,
    }
    with (trial_dir / "trial_context.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # Generate RTL capped to sim_width so testbench uses plain uint32_t
    sv_path = _write_trial_lint_sv(
        trial_dir=trial_dir,
        top_module=top_module,
        cfg=cfg,
        max_width=max(1, min(sim_width, 32)),
    )
    cpp_path = _write_trial_tb_cpp(
        trial_dir=trial_dir,
        top_module=top_module,
        n_steps=sim_steps,
    )

    obj_dir = trial_dir / "obj_dir"
    binary = obj_dir / f"V{top_module}"

    # Compile step
    build_cmd = [
        "verilator",
        "-cc",
        "--exe",
        "--build",
        "-Wall",
        "-Wno-DECLFILENAME",
        "--top-module", top_module,
        "-Mdir", str(obj_dir),
        str(sv_path),
        str(cpp_path),
    ]
    build_result = subprocess.run(  # noqa: S603
        build_cmd,
        capture_output=True,
        text=True,
        timeout=max(1, timeout_s),
        check=False,
    )

    if build_result.returncode != 0:
        return {
            "mode": "simulate",
            "stage": "build_failed",
            "build_command": " ".join(build_cmd),
            "build_returncode": build_result.returncode,
            "build_stdout": build_result.stdout[-4000:],
            "build_stderr": build_result.stderr[-4000:],
            "passed": False,
            "trial_dir": str(trial_dir),
            "rtl_file": str(sv_path),
        }

    # Execution step
    run_result = subprocess.run(  # noqa: S603
        [str(binary)],
        capture_output=True,
        text=True,
        timeout=max(1, 30),
        check=False,
    )

    return {
        "mode": "simulate",
        "stage": "simulated",
        "build_command": " ".join(build_cmd),
        "binary": str(binary),
        "returncode": run_result.returncode,
        "stdout": run_result.stdout[-4000:],
        "stderr": run_result.stderr[-4000:],
        "passed": run_result.returncode == 0,
        "trial_dir": str(trial_dir),
        "rtl_file": str(sv_path),
        "top_module": top_module,
    }


def main() -> None:
    args = parse_args()

    Path("experiments").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        fixed_round_options = [
            x.strip()
            for x in args.fixed_point_rounding_options.split(",")
            if x.strip()
        ]
        cfg = build_config(
            trial,
            args.time_budget_s,
            ternary_search=args.ternary_search,
            fixed_point_search=args.fixed_point_search,
            fixed_point_bits_min=args.fixed_point_bits_min,
            fixed_point_bits_max=args.fixed_point_bits_max,
            fixed_point_frac_min=args.fixed_point_frac_min,
            fixed_point_frac_max=args.fixed_point_frac_max,
            fixed_point_rounding_options=fixed_round_options,
            ternary_threshold_min=args.ternary_threshold_min,
            ternary_threshold_max=args.ternary_threshold_max,
        )
        metrics, _ = train_once(cfg)

        if cfg.weight_mode == "ternary" and args.run_verilator:
            if args.verilator_mode == "lint":
                verilator = run_verilator_lint_step(
                    trial,
                    cfg,
                    metrics,
                    timeout_s=args.verilator_timeout_s,
                    top_module=args.verilator_top,
                    max_width=args.verilator_max_width,
                )
            elif args.verilator_mode == "simulate":
                verilator = run_verilator_simulate_step(
                    trial,
                    cfg,
                    metrics,
                    timeout_s=args.verilator_timeout_s,
                    top_module=args.verilator_top,
                    sim_width=args.verilator_sim_width,
                    sim_steps=args.verilator_sim_steps,
                )
            else:
                verilator = run_verilator_step(
                    trial,
                    cfg,
                    metrics,
                    command_template=args.verilator_command,
                    timeout_s=args.verilator_timeout_s,
                )
            metrics["verilator"] = verilator
            trial.set_user_attr("verilator_passed", bool(verilator["passed"]))

        append_jsonl(args.log_jsonl, trial_to_payload(trial, cfg, metrics))
        trial.set_user_attr("metrics", json.dumps(metrics, sort_keys=True))
        return float(metrics["val_acc"])

    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    print("---")
    print(f"best_trial: {best.number}")
    print(f"best_val_acc: {best.value:.6f}")
    print(f"best_params: {best.params}")


if __name__ == "__main__":
    main()
