from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import TypedDict


class CurriculumStage(TypedDict):
    name: str
    action_noise: float
    wind_std: float
    act_noise: float
    latency: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Disturbance curriculum training for motor-mode informed Dreamer")
    p.add_argument("--output-root", type=str, default="artifacts/sim/motor_curriculum")
    p.add_argument("--feature-dim", type=int, default=79)
    p.add_argument("--num-sequences", type=int, default=64)
    p.add_argument("--sequence-len", type=int, default=16)
    p.add_argument("--epochs-per-stage", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--horizon", type=int, default=8)
    return p.parse_args()


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    print(proc.stdout, end="")
    return proc.stdout


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    py = sys.executable
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stages: list[CurriculumStage] = [
        {"name": "stage0_calm", "action_noise": 0.18, "wind_std": 0.0, "act_noise": 0.0, "latency": 0},
        {"name": "stage1_wind", "action_noise": 0.20, "wind_std": 0.5, "act_noise": 0.04, "latency": 0},
        {"name": "stage2_gust", "action_noise": 0.22, "wind_std": 1.0, "act_noise": 0.08, "latency": 1},
        {"name": "stage3_noisy", "action_noise": 0.24, "wind_std": 1.0, "act_noise": 0.12, "latency": 2},
    ]

    rows: list[dict[str, str]] = []

    for i, st in enumerate(stages):
        stage_dir = out_root / st["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        dataset = stage_dir / "sim_rollouts.h5"

        build_cmd = [
            py,
            "-m",
            "world_model_experiments.build_sim_rollout_dataset",
            "--output",
            str(dataset),
            "--num-sequences",
            str(args.num_sequences),
            "--sequence-len",
            str(args.sequence_len),
            "--feature-dim",
            str(args.feature_dim),
            "--action-noise",
            str(st["action_noise"]),
            "--wind-std",
            str(st["wind_std"]),
            "--actuation-noise-std",
            str(st["act_noise"]),
            "--latency-steps",
            str(st["latency"]),
            "--seed",
            str(args.seed + i),
        ]
        run_cmd(build_cmd)

        train_cmd = [
            py,
            "-m",
            "world_model_experiments.train_informed_dreamer",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(stage_dir),
            "--epochs",
            str(args.epochs_per_stage),
            "--batch-size",
            str(args.batch_size),
            "--embed-dim",
            str(args.embed_dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--horizon",
            str(args.horizon),
            "--seed",
            str(args.seed + i),
            "--use-motor-commands",
        ]
        run_cmd(train_cmd)

        eval_cmd = [
            py,
            "-m",
            "world_model_experiments.evaluate_closed_loop_motor",
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(stage_dir / "informed_dreamer_best.pt"),
            "--episodes",
            "8",
            "--horizon",
            str(args.horizon),
            "--use-motor-commands",
        ]
        eval_out = run_cmd(eval_cmd)

        rows.append(
            {
                "stage": st["name"],
                "dataset": str(dataset),
                "checkpoint": str(stage_dir / "informed_dreamer_best.pt"),
                "wind_std": str(st["wind_std"]),
                "actuation_noise_std": str(st["act_noise"]),
                "latency_steps": str(st["latency"]),
                "closed_loop_eval_tail": eval_out.strip().splitlines()[-1] if eval_out.strip() else "",
            }
        )

    write_csv(out_root / "curriculum_summary.csv", rows)
    print(f"wrote: {out_root / 'curriculum_summary.csv'}")


if __name__ == "__main__":
    main()
