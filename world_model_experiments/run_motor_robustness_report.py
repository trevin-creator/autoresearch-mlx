from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run disturbance robustness report from per-seed checkpoints")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint-root", type=str, required=True)
    p.add_argument("--output-root", type=str, default="artifacts/sim/motor_robustness")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--horizon", type=int, default=8)
    return p.parse_args()


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    print(proc.stdout, end="")
    return proc.stdout


def parse_robust_eval(output: str) -> dict[str, dict[str, float]]:
    for line in reversed(output.splitlines()):
        if line.startswith("robust_eval "):
            payload = line[len("robust_eval ") :].strip()
            parsed = json.loads(payload)
            out: dict[str, dict[str, float]] = {}
            for scenario, metrics in parsed.items():
                out[str(scenario)] = {str(k): float(v) for k, v in metrics.items()}
            return out
    raise RuntimeError("Could not parse robust_eval output")


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    per_seed: dict[int, dict[str, dict[str, float]]] = {}

    for seed in seeds:
        ckpt = Path(args.checkpoint_root) / f"seed_{seed}" / "informed_dreamer_best.pt"
        cmd: list[str] = [
            py,
            "-m",
            "world_model_experiments.evaluate_motor_robustness",
            "--dataset",
            args.dataset,
            "--checkpoint",
            str(ckpt),
            "--episodes",
            str(args.episodes),
            "--horizon",
            str(args.horizon),
            "--use-motor-commands",
            "--seed",
            str(seed),
        ]
        out = run_cmd(cmd)
        per_seed[seed] = parse_robust_eval(out)

    scenarios = sorted(next(iter(per_seed.values())).keys())
    metric_keys = sorted(next(iter(next(iter(per_seed.values())).values())).keys())

    rows: list[list[str]] = []
    for seed in seeds:
        for scenario in scenarios:
            rows.append([str(seed), scenario, *[f"{per_seed[seed][scenario][k]:.9f}" for k in metric_keys]])

    csv_path = out_root / "motor_robustness_by_seed.csv"
    write_csv(csv_path, ["seed", "scenario", *metric_keys], rows)

    report = out_root / "motor_robustness_report.md"
    lines = [
        "# Motor Robustness Report",
        "",
        f"- dataset: {args.dataset}",
        f"- checkpoint_root: {args.checkpoint_root}",
        f"- seeds: {','.join(str(s) for s in seeds)}",
        "",
        "## Scenario Means Across Seeds",
    ]

    for scenario in scenarios:
        lines.append(f"### {scenario}")
        for k in metric_keys:
            vals = [per_seed[s][scenario][k] for s in seeds]
            lines.append(f"- {k}: mean={statistics.mean(vals):.9f}, std={statistics.pstdev(vals):.9f}")
        lines.append("")

    lines.extend([
        "## Artifacts",
        "- motor_robustness_by_seed.csv",
    ])
    report.write_text("\n".join(lines) + "\n")

    print(f"wrote: {csv_path}")
    print(f"wrote: {report}")


if __name__ == "__main__":
    main()
