"""
rigor.py — significance-gated keep/discard for the autoresearch loop.

A single training run is noisy: re-running the *same* train.py moves val_bpb by
~0.03 here (nondeterministic GPU reductions, even with the seed pinned). Many
changes the loop accepts improve by less than that, so it chases noise. And
because it only keeps a run that dips below the running best and never re-checks,
the recorded curve is an optimistic running-minimum that regresses on honest
re-evaluation.

This replaces "run once, eyeball the delta" with "run a few seeds, keep only if
the change is better with high confidence." It also remembers every config it
scored (hash of train.py), so an identical proposal is never re-run.

    uv run rigor.py run "halve the batch size"        # score train.py vs best
    uv run rigor.py run "..." --seeds 5 --confidence 0.9
    uv run rigor.py best                              # current best
    uv run rigor.py log                              # every scored config

Samples and decisions land in rigor_ledger.jsonl. Nothing here edits train.py,
touches git, or changes evaluate_bpb — it only decides.
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"
LEDGER = ROOT / "rigor_ledger.jsonl"
TRAIN_CMD = ["uv", "run", "train.py"]
VAL_RE = re.compile(r"^val_bpb:\s+([\d.]+)", re.M)


def train_hash():
    return hashlib.sha1(TRAIN.read_bytes()).hexdigest()[:7]


def run_once():
    """Run one training run, echo it live, return final val_bpb (None on crash)."""
    proc = subprocess.Popen(TRAIN_CMD, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = []
    for line in proc.stdout:
        sys.stdout.write(line)
        out.append(line)
    proc.wait()
    m = VAL_RE.search("".join(out))
    return float(m.group(1)) if proc.returncode == 0 and m else None


def load_ledger():
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]


def record(h, desc, samples, status, p):
    entry = {"hash": h, "desc": desc, "samples": samples,
             "mean": statistics.mean(samples) if samples else 0.0,
             "std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
             "status": status, "p_better": round(p, 4)}
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def best_entry(ledger):
    keeps = [e for e in ledger if e["status"] == "keep"]
    return min(keeps, key=lambda e: e["mean"], default=None)


def prob_improvement(best, cand, trials=20000):
    """Bootstrap P(mean(cand) < mean(best)) — fraction of resamples the candidate wins."""
    rng = random.Random(1234)  # fixed so a verdict is reproducible
    nb, nc = len(best), len(cand)
    wins = 0
    for _ in range(trials):
        b = sum(best[rng.randrange(nb)] for _ in range(nb)) / nb
        c = sum(cand[rng.randrange(nc)] for _ in range(nc)) / nc
        wins += c < b
    return wins / trials


def score(desc, seeds, confidence):
    ledger = load_ledger()
    h = train_hash()
    prior = next((e for e in ledger if e["hash"] == h), None)
    if prior:
        print(f"already scored {h}: {prior['status']} (mean {prior['mean']:.6f}) — {prior['desc']}")
        print("skipping (never-repeat). change train.py to try something new.")
        return

    best = best_entry(ledger)
    samples = []
    for i in range(seeds):
        v = run_once()
        if v is None:
            record(h, desc, samples, "crash", 0.0)
            print(f"\nCRASH {h}: training did not finish — logged.")
            return
        samples.append(v)
        print(f"  seed {i + 1}/{seeds}: val_bpb {v:.6f}")
        if best and i == 0 and v >= best["mean"]:  # clear loser, don't spend more seeds
            record(h, desc, samples, "discard", 0.0)
            print(f"\nDISCARD {h}: first run {v:.6f} >= best mean {best['mean']:.6f}.")
            return

    if best is None:  # first config is the baseline
        e = record(h, desc, samples, "keep", 1.0)
        print(f"\nBASELINE {h}: {e['mean']:.6f} +/- {e['std']:.6f} ({seeds} seeds)")
        return

    p = prob_improvement(best["samples"], samples)
    keep = p >= confidence
    status = "keep" if keep else "discard"
    e = record(h, desc, samples, status, p)
    print(f"\n{status.upper()} {h}: mean {e['mean']:.6f} vs best {best['mean']:.6f} "
          f"(delta {e['mean'] - best['mean']:+.6f}), P(better)={p:.2f}, need >={confidence:.2f}")
    print(f"tsv: {h}\t{e['mean']:.6f}\t-\t{status}\t{desc}")


def main():
    ap = argparse.ArgumentParser(description="significance-gated keep/discard for the autoresearch loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score the current train.py against the best")
    r.add_argument("desc", help="short description of the change")
    r.add_argument("--seeds", type=int, default=3, help="training runs to average (default 3)")
    r.add_argument("--confidence", type=float, default=0.95, help="P(better) needed to keep (default 0.95)")
    sub.add_parser("best", help="show the current best config")
    sub.add_parser("log", help="show every scored config")
    a = ap.parse_args()

    if a.cmd == "run":
        score(a.desc, a.seeds, a.confidence)
    elif a.cmd == "best":
        b = best_entry(load_ledger())
        print(f"best {b['hash']}: {b['mean']:.6f} +/- {b['std']:.6f} — {b['desc']}" if b else "no baseline yet.")
    elif a.cmd == "log":
        for e in load_ledger():
            print(f"{e['hash']}  {e['mean']:.6f}  {e['status']:8s}  p={e['p_better']:.2f}  {e['desc']}")


if __name__ == "__main__":
    main()
