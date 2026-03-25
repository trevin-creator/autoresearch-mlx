from __future__ import annotations

import argparse
import ast
import csv
import statistics
import subprocess
import sys
from pathlib import Path

from world_model_experiments._errors import ERR_SEEDS_EMPTY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multiseed motor-mode informed Dreamer training and report")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output-root", type=str, default="artifacts/sim/motor_multiseed")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--closed-loop-episodes", type=int, default=8)
    return p.parse_args()


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    print(proc.stdout, end="")
    return proc.stdout


def parse_prefixed_dict(output: str, prefix: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        if line.startswith(prefix + " "):
            raw = line[len(prefix) + 1 :].strip()
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return {str(k): float(v) for k, v in parsed.items()}
    raise RuntimeError(f"Could not parse {prefix} metrics")


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError(ERR_SEEDS_EMPTY)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    eval_rows: list[dict[str, float]] = []
    closed_rows: list[dict[str, float]] = []

    for seed in seeds:
        run_dir = out_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "informed_dreamer_best.pt"

        train_cmd = [
            py,
            "-m",
            "world_model_experiments.train_informed_dreamer",
            "--dataset",
            args.dataset,
            "--output-dir",
            str(run_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--embed-dim",
            str(args.embed_dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--horizon",
            str(args.horizon),
            "--seed",
            str(seed),
            "--use-motor-commands",
        ]
        run_cmd(train_cmd)

        eval_cmd = [
            py,
            "-m",
            "world_model_experiments.evaluate_informed_dreamer",
            "--dataset",
            args.dataset,
            "--checkpoint",
            str(ckpt),
            "--use-motor-commands",
        ]
        eval_out = run_cmd(eval_cmd)
        eval_rows.append(parse_prefixed_dict(eval_out, "informed_eval"))

        cl_cmd = [
            py,
            "-m",
            "world_model_experiments.evaluate_closed_loop_motor",
            "--dataset",
            args.dataset,
            "--checkpoint",
            str(ckpt),
            "--episodes",
            str(args.closed_loop_episodes),
            "--horizon",
            str(args.horizon),
            "--seed",
            str(seed),
            "--use-motor-commands",
        ]
        cl_out = run_cmd(cl_cmd)
        closed_rows.append(parse_prefixed_dict(cl_out, "closed_loop_eval"))

    eval_keys = sorted(eval_rows[0].keys()) if eval_rows else []
    closed_keys = sorted(closed_rows[0].keys()) if closed_rows else []

    eval_csv_rows = [[str(seeds[i]), *[f"{eval_rows[i][k]:.9f}" for k in eval_keys]] for i in range(len(seeds))]
    cl_csv_rows = [[str(seeds[i]), *[f"{closed_rows[i][k]:.9f}" for k in closed_keys]] for i in range(len(seeds))]

    write_csv(out_root / "motor_open_loop_eval.csv", ["seed", *eval_keys], eval_csv_rows)
    write_csv(out_root / "motor_closed_loop_eval.csv", ["seed", *closed_keys], cl_csv_rows)

    summary_path = out_root / "motor_report.md"
    lines = [
        "# Motor-Mode Multiseed Report",
        "",
        f"- dataset: {args.dataset}",
        f"- seeds: {','.join(str(s) for s in seeds)}",
        f"- epochs: {args.epochs}",
        f"- batch_size: {args.batch_size}",
        "",
        "## Open-Loop Mean Across Seeds",
    ]
    for k in eval_keys:
        vals = [row[k] for row in eval_rows]
        lines.append(f"- {k}: mean={statistics.mean(vals):.9f}, std={statistics.pstdev(vals):.9f}")

    lines.append("")
    lines.append("## Closed-Loop Mean Across Seeds")
    for k in closed_keys:
        vals = [row[k] for row in closed_rows]
        lines.append(f"- {k}: mean={statistics.mean(vals):.9f}, std={statistics.pstdev(vals):.9f}")

    lines.extend([
        "",
        "## Artifact Files",
        "- motor_open_loop_eval.csv",
        "- motor_closed_loop_eval.csv",
    ])
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"wrote: {out_root / 'motor_open_loop_eval.csv'}")
    print(f"wrote: {out_root / 'motor_closed_loop_eval.csv'}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
