"""
rigor.py — significance-gated keep/discard for any noisy experiment loop.

A single run of an experiment is noisy: re-running the *same* code moves the
metric (here, val_bpb shifts by ~0.03 from nondeterministic GPU reductions even
with the seed pinned). A loop that keeps whatever dips past the running best
chases that noise, and its recorded curve is an optimistic running-minimum that
regresses on honest re-evaluation.

This replaces "run once, eyeball the delta" with "run a few times, keep only if
the change is better with high confidence" (bootstrap). It remembers every
config it scored (hash of the watched files), so an identical proposal is never
re-run.

Works for any experiment that is a command printing `<metric>: <number>` (or
`<metric>=<number>`); the last such line is the score. Defaults reproduce the
autoresearch setup (uv run train.py, minimize val_bpb, watch train.py):

    uv run rigor.py run "halve the batch size"
    uv run rigor.py run "..." --seeds 5 --confidence 0.9
    uv run rigor.py best
    uv run rigor.py log

Several metrics: the first --metric decides keep/discard; the rest are
guardrails — a change is discarded if it is confidently worse on any of them.
Append :max for higher-is-better (default :min). Options go before the
subcommand; use a separate ledger per study:

    python rigor.py --cmd "python bench.py" --metric throughput:max \\
        --metric p99_latency --metric memory_mb \\
        --files bench.py kernel.c --ledger bench_ledger.jsonl run "unroll loop"

Run i gets seed i through the SEED env var (--seed-env), so every config is run
on the same seeds 0..n-1. With at least 5 seeds shared with the best
(--seeds 5), runs pair up by seed and an exact sign-flip test on the per-seed
differences decides, which cancels seed-to-seed variance. With fewer (the
default 3) or older ledger entries without seeds, it uses the unpaired
bootstrap. Three paired seeds can never show 95% confidence honestly (the
best possible is 1 - 1/8).

Paths and the command are relative to the current directory. Nothing here
edits the watched files, touches git, or changes how a metric is computed —
it only decides.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

NUMBER = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
AUTO_MARGIN_STDS = 3  # auto early-discard margin = this many stdevs of the best's runs
MIN_PAIRED_SEEDS = 5  # fewest shared seeds for the paired test (max confidence 1 - 1/2^5 = 0.97)


@dataclass(frozen=True)
class Metric:
    name: str
    maximize: bool = False

    def key(self, v):
        """Map a value so that lower is always better."""
        return -v if self.maximize else v


@dataclass(frozen=True)
class Study:
    cmd: list
    metrics: list  # metrics[0] decides keep/discard; the rest are guardrails
    files: list  # paths whose contents identify a config
    ledger: Path
    seed_env: str = "SEED"  # env var holding the run's seed ("" = don't set one)

    @property
    def primary(self):
        return self.metrics[0]


def metric_spec(spec):
    name, _, direction = spec.partition(":")
    if not name or direction not in ("", "min", "max"):
        raise argparse.ArgumentTypeError(f"bad metric '{spec}': use NAME, NAME:min or NAME:max")
    return Metric(name, direction == "max")


def config_hash(study):
    # one file hashes exactly as before, so existing ledgers keep their never-repeat memory
    data = b"\0".join(Path(f).read_bytes() for f in study.files)
    return hashlib.sha1(data).hexdigest()[:7]


def parse_metric(output, metric):
    pat = re.compile(rf"^\s*{re.escape(metric)}\s*[:=]\s*{NUMBER}", re.M)
    found = pat.findall(output)
    return float(found[-1]) if found else None


def run_once(study, seed=None):
    """Run the experiment once, echo it live, return {metric: value} (None on crash)."""
    env = dict(os.environ)
    if study.seed_env and seed is not None:
        env[study.seed_env] = str(seed)
    proc = subprocess.Popen(study.cmd, text=True, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = []
    for line in proc.stdout:
        sys.stdout.write(line)
        out.append(line)
    proc.wait()
    if proc.returncode != 0:
        return None
    text = "".join(out)
    values = {m.name: parse_metric(text, m.name) for m in study.metrics}
    return None if None in values.values() else values


def load_ledger(study):
    if not study.ledger.exists():
        return []
    return [json.loads(x) for x in study.ledger.read_text().splitlines() if x.strip()]


def record(study, h, desc, samples, seeds, status, p, regressed=()):
    """samples: {metric: [values]}, seeds[i] is run i's seed. Primary stays in samples/mean/std."""
    prim = samples.get(study.primary.name, [])
    entry = {"hash": h, "desc": desc, "samples": prim, "seeds": list(seeds),
             "mean": statistics.mean(prim) if prim else 0.0,
             "std": statistics.pstdev(prim) if len(prim) > 1 else 0.0,
             "status": status, "p_better": round(p, 4)}
    extra = {m.name: samples[m.name] for m in study.metrics[1:] if samples.get(m.name)}
    if extra:
        entry["extra"] = extra
    if regressed:
        entry["regressed"] = list(regressed)
    with study.ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def best_entry(study, ledger):
    keeps = [e for e in ledger if e["status"] == "keep"]
    return min(keeps, key=lambda e: study.primary.key(e["mean"]), default=None)


def prob_improvement(best, cand, trials=20000):
    """Bootstrap P(mean(cand) < mean(best)) on lower-is-better values."""
    rng = random.Random(1234)  # fixed so a verdict is reproducible
    nb, nc = len(best), len(cand)
    wins = 0
    for _ in range(trials):
        b = sum(best[rng.randrange(nb)] for _ in range(nb)) / nb
        c = sum(cand[rng.randrange(nc)] for _ in range(nc)) / nc
        wins += c < b
    return wins / trials


def sign_flip_p_better(diffs):
    """Exact paired sign-flip test on per-seed differences (negative = candidate better).
    Returns 1 - p, where p = share of the 2^n sign flips whose mean is <= the observed one.
    With n pairs it can't exceed 1 - 1/2^n, so it never overstates what few seeds show."""
    # ponytail: enumerates all 2^n flips; fine for n <= ~16 seeds, sample flips beyond that
    obs = sum(diffs)
    tol = 1e-12 * max(1.0, *(abs(d) for d in diffs))
    flips = itertools.product((1, -1), repeat=len(diffs))
    as_good = sum(sum(s * d for s, d in zip(f, diffs)) <= obs + tol for f in flips)
    return 1 - as_good / 2 ** len(diffs)


def paired_diffs(best_vals, best_seeds, cand_vals, cand_seeds):
    """cand - best at each seed both ran."""
    by_seed = dict(zip(best_seeds or [], best_vals))
    return [c - by_seed[s] for s, c in zip(cand_seeds or [], cand_vals) if s in by_seed]


def p_better(metric, best_vals, cand_vals, best_seeds=None, cand_seeds=None):
    """(P(cand better than best), test used). Paired sign-flip test on shared seeds when
    there are enough of them (cancels seed-to-seed variance); otherwise unpaired bootstrap."""
    k = metric.key
    diffs = paired_diffs([k(x) for x in best_vals], best_seeds, [k(x) for x in cand_vals], cand_seeds)
    if len(diffs) >= MIN_PAIRED_SEEDS:
        return sign_flip_p_better(diffs), "paired"
    return prob_improvement([k(x) for x in best_vals], [k(x) for x in cand_vals]), "unpaired"


def regressions(study, best, samples, seeds, confidence):
    """Guardrail metrics the candidate is confidently worse on than the best."""
    bad = []
    for m in study.metrics[1:]:
        ref = best.get("extra", {}).get(m.name)
        if not ref:
            print(f"  guardrail {m.name}: best has no samples, not checked")
            continue
        # P(candidate worse) = P(best better than candidate)
        p_worse, _ = p_better(m, samples[m.name], ref, seeds, best.get("seeds"))
        print(f"  guardrail {m.name}: {statistics.mean(samples[m.name]):.6g} vs best "
              f"{statistics.mean(ref):.6g}, P(worse)={p_worse:.2f}")
        if p_worse >= confidence:
            bad.append(m.name)
    return bad


def early_margin_for(best, early_margin):
    """Explicit margin if given; else scale-free: a few stdevs of the best's own runs."""
    if early_margin is not None:
        return early_margin
    sd = statistics.stdev(best["samples"]) if len(best["samples"]) > 1 else 0.0
    return AUTO_MARGIN_STDS * sd if sd > 0 else None  # no noise estimate: never discard early


def is_clear_loser(study, best, v, margin):
    key = study.primary.key
    return margin is not None and key(v) >= key(best["mean"]) + margin


def collect(study, h, desc, seeds, best, margin):
    """Run `seeds` times with seeds 0..seeds-1 (the same for every config, so runs pair up).
    Returns (samples, seeds_used), or None if it already logged a crash/early discard."""
    pname = study.primary.name
    samples = {m.name: [] for m in study.metrics}
    used = []
    for i in range(seeds):
        vals = run_once(study, i)
        if vals is None:
            record(study, h, desc, samples, used, "crash", 0.0)
            names = ", ".join(m.name for m in study.metrics)
            print(f"\nCRASH {h}: run failed or didn't print every metric ({names}) — logged.")
            return None
        for name, v in vals.items():
            samples[name].append(v)
        used.append(i)
        v = vals[pname]
        print(f"  run {i + 1}/{seeds} (seed {i}): " + "  ".join(f"{k} {x:.6g}" for k, x in vals.items()))
        # clear loser: worse than best by more than run-to-run noise, don't spend more runs
        if best and i == 0 and seeds > 1 and is_clear_loser(study, best, v, margin):
            record(study, h, desc, samples, used, "discard", 0.0)
            print(f"\nDISCARD {h}: first run {v:.6f} worse than best mean "
                  f"{best['mean']:.6f} by more than margin {margin:.6g}.")
            return None
    return samples, used


def score(study, desc, seeds, confidence, early_margin=None):
    ledger = load_ledger(study)
    h = config_hash(study)
    # crashes may be transient (OOM, killed process), so they don't block a retry
    prior = next((e for e in ledger if e["hash"] == h and e["status"] != "crash"), None)
    if prior:
        print(f"already scored {h}: {prior['status']} (mean {prior['mean']:.6f}) — {prior['desc']}")
        print("skipping (never-repeat). change the watched files to try something new.")
        return

    best = best_entry(study, ledger)
    margin = early_margin_for(best, early_margin) if best else None
    got = collect(study, h, desc, seeds, best, margin)
    if got is None:
        return
    samples, used = got

    if best is None:  # first config is the baseline
        e = record(study, h, desc, samples, used, "keep", 1.0)
        print(f"\nBASELINE {h}: {e['mean']:.6f} +/- {e['std']:.6f} ({seeds} runs)")
        return

    p, test = p_better(study.primary, best["samples"], samples[study.primary.name],
                       best.get("seeds"), used)
    print(f"  test: {test}" + ("" if test == "paired" else
          f" (paired needs >= {MIN_PAIRED_SEEDS} seeds shared with the best; use --seeds 5)"))
    bad = regressions(study, best, samples, used, confidence)
    status = "keep" if p >= confidence and not bad else "discard"
    e = record(study, h, desc, samples, used, status, p, bad)
    print(f"\n{status.upper()} {h}: mean {e['mean']:.6f} vs best {best['mean']:.6f} "
          f"(delta {e['mean'] - best['mean']:+.6f}), P(better)={p:.2f}, need >={confidence:.2f}"
          + (f"; worse on {', '.join(bad)}" if bad else ""))
    print(f"tsv: {h}\t{e['mean']:.6f}\t-\t{status}\t{desc}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="significance-gated keep/discard for noisy experiments")
    ap.add_argument("--cmd", default="uv run train.py", help="experiment command (default: uv run train.py)")
    ap.add_argument("--metric", type=metric_spec, action="append", dest="metrics",
                    help="NAME[:min|:max] printed as '<name>: <value>'. Repeat for guardrails; "
                         "the first decides keep/discard (default val_bpb:min)")
    ap.add_argument("--files", nargs="+", default=["train.py"],
                    help="files whose contents identify a config (default train.py)")
    ap.add_argument("--ledger", default="rigor_ledger.jsonl", help="ledger path (one per study)")
    ap.add_argument("--seed-env", default="SEED",
                    help="env var set to the run's seed (0, 1, 2, ...); '' to not set one (default SEED)")
    sub = ap.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run", help="score the current config against the best")
    r.add_argument("desc", help="short description of the change")
    r.add_argument("--seeds", type=int, default=3, help="runs to average (default 3)")
    r.add_argument("--confidence", type=float, default=0.95, help="P(better) needed to keep (default 0.95)")
    r.add_argument("--early-margin", type=float, default=None,
                   help="discard after one run only if worse than best by this much on the first "
                        f"metric (default: {AUTO_MARGIN_STDS}x stdev of the best's runs)")
    sub.add_parser("best", help="show the current best config")
    sub.add_parser("log", help="show every scored config")
    return ap.parse_args(argv)


def fmt_extra(e):
    return "  ".join(f"{k}={statistics.mean(v):.6g}" for k, v in e.get("extra", {}).items())


def main():
    a = parse_args()
    study = Study(cmd=shlex.split(a.cmd), metrics=a.metrics or [Metric("val_bpb")],
                  files=[Path(f) for f in a.files], ledger=Path(a.ledger), seed_env=a.seed_env)
    if a.sub == "run":
        score(study, a.desc, a.seeds, a.confidence, a.early_margin)
    elif a.sub == "best":
        b = best_entry(study, load_ledger(study))
        print(f"best {b['hash']}: {b['mean']:.6f} +/- {b['std']:.6f} {fmt_extra(b)} — {b['desc']}"
              if b else "no baseline yet.")
    elif a.sub == "log":
        for e in load_ledger(study):
            print(f"{e['hash']}  {e['mean']:.6f}  {e['status']:8s}  p={e['p_better']:.2f}  "
                  f"{fmt_extra(e)}  {e['desc']}")


if __name__ == "__main__":
    main()
