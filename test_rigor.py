"""Self-check for rigor.py's keep/discard rules, no real runs needed: uv run test_rigor.py"""

import sys
import tempfile
from pathlib import Path

import rigor
from rigor import Metric

REAL_RUN_ONCE = rigor.run_once  # tests below replace rigor.run_once with fakes


def make_study(*metrics):
    tmp = Path(tempfile.mkdtemp())
    return rigor.Study(cmd=[], metrics=list(metrics) or [Metric("m")],
                       files=[tmp / "exp.py"], ledger=tmp / "ledger.jsonl")


def run(study, code, results, margin=0.03, seeds=3):
    """results: per run, a float (single metric), a {metric: value} dict, or None (crash)."""
    study.files[0].write_text(code)
    name = study.primary.name
    it = iter(r if r is None or isinstance(r, dict) else {name: r} for r in results)
    rigor.run_once = lambda _study, _seed=None: next(it)
    rigor.score(study, "t", seeds, 0.95, margin)
    return rigor.load_ledger(study)[-1]


def check_minimize():
    s = make_study()
    assert run(s, "a", [1.50, 1.50, 1.50])["status"] == "keep"  # baseline
    # first run slightly worse (within noise) must NOT be discarded early
    assert len(run(s, "b", [1.51, 1.40, 1.40])["samples"]) == 3
    # first run worse by more than the margin is discarded after one run
    e = run(s, "c", [1.60])
    assert e["status"] == "discard" and len(e["samples"]) == 1, e
    # a crash doesn't block retrying the same config...
    assert run(s, "d", [None])["status"] == "crash"
    n = len(rigor.load_ledger(s))
    run(s, "d", [1.45, 1.45, 1.45])
    assert len(rigor.load_ledger(s)) == n + 1
    # ...but a scored config is never re-run
    run(s, "d", [1.0, 1.0, 1.0])
    assert len(rigor.load_ledger(s)) == n + 1


def check_maximize():
    s = make_study(Metric("m", maximize=True))
    run(s, "a", [100, 101, 99])
    assert run(s, "b", [120, 121, 119])["status"] == "keep"  # higher wins
    assert rigor.best_entry(s, rigor.load_ledger(s))["mean"] == 120
    # auto margin = 3 * stdev(best) = 3: 118 is within noise, 110 is a clear loser
    assert len(run(s, "c", [118, 118, 118], margin=None)["samples"]) == 3
    assert len(run(s, "d", [110], margin=None)["samples"]) == 1


def check_auto_margin_needs_noise():
    s = make_study()
    run(s, "a", [1.0, 1.0, 1.0])  # zero spread: no noise estimate, so no early discard
    assert len(run(s, "b", [5.0, 5.0, 5.0], margin=None)["samples"]) == 3


def check_guardrails():
    s = make_study(Metric("loss"), Metric("thr", maximize=True), Metric("mem"))
    base = [{"loss": 2.0 + d, "thr": 100 + d, "mem": 50 + d} for d in (-0.01, 0, 0.01)]
    e = run(s, "a", base)
    assert e["status"] == "keep" and set(e["extra"]) == {"thr", "mem"}

    # better loss but clearly lower throughput: guardrail rejects it
    worse_thr = [{"loss": 1.5 + d, "thr": 80 + d, "mem": 50 + d} for d in (-0.01, 0, 0.01)]
    e = run(s, "b", worse_thr)
    assert e["status"] == "discard" and e["regressed"] == ["thr"], e

    # better loss, guardrails within noise: kept
    ok = [{"loss": 1.5 + d, "thr": 100 - d, "mem": 50 + d} for d in (-0.01, 0, 0.01)]
    assert run(s, "c", ok)["status"] == "keep"

    # a real run that doesn't print every metric counts as a crash
    prints_two = rigor.Study(cmd=[sys.executable, "-c", "print('loss: 1\\nthr: 2')"],
                             metrics=s.metrics, files=s.files, ledger=s.ledger)
    assert REAL_RUN_ONCE(prints_two) is None
    prints_all = rigor.Study(cmd=[sys.executable, "-c", "print('loss: 1\\nthr = 2\\nmem: 3e1')"],
                             metrics=s.metrics, files=s.files, ledger=s.ledger)
    assert REAL_RUN_ONCE(prints_all) == {"loss": 1.0, "thr": 2.0, "mem": 30.0}


def check_old_ledger_best_without_guardrail_samples():
    s = make_study(Metric("loss"), Metric("mem"))
    s.ledger.write_text('{"hash": "x", "desc": "old", "samples": [2.0, 2.01, 1.99], '
                        '"mean": 2.0, "std": 0.008, "status": "keep", "p_better": 1.0}\n')
    better = [{"loss": 1.5 + d, "mem": 999} for d in (-0.01, 0, 0.01)]
    assert run(s, "a", better)["status"] == "keep"  # guardrail skipped, primary decides


def check_sign_flip():
    # n consistent improvements: best possible confidence is 1 - 1/2^n, never more
    assert rigor.sign_flip_p_better([-1, -2, -3]) == 1 - 1 / 8
    assert rigor.sign_flip_p_better([-1, -2, -3, -4, -5]) == 1 - 1 / 32
    assert rigor.sign_flip_p_better([0, 0, 0, 0, 0]) == 0  # no difference: no evidence
    assert rigor.sign_flip_p_better([1, 2, 3, 4, 5]) == 0  # consistently worse


def check_paired_cancels_seed_variance():
    # big seed-to-seed spread, small but consistent improvement on every seed
    s = make_study()
    seed_effect = [1.0, 1.2, 1.4, 1.6, 1.8]
    run(s, "a", seed_effect, seeds=5)
    e = run(s, "b", [x - 0.01 for x in seed_effect], seeds=5, margin=None)
    assert e["seeds"] == [0, 1, 2, 3, 4] and e["status"] == "keep", e  # paired sees it
    # unpaired would not: the spread swamps a 0.01 shift
    assert rigor.prob_improvement(seed_effect, [x - 0.01 for x in seed_effect]) < 0.95


def check_seed_reaches_command():
    s = rigor.Study(cmd=[sys.executable, "-c", "import os; print('m:', os.environ['SEED'])"],
                    metrics=[Metric("m")], files=[], ledger=Path("unused"))
    assert REAL_RUN_ONCE(s, 7) == {"m": 7.0}


def check_parse():
    out = "step 1\nloss: 3.2\nloss = 2.5e-1\nval_loss: 9\n"
    assert rigor.parse_metric(out, "loss") == 0.25  # last match wins
    assert rigor.parse_metric("acc: -.5\n", "acc") == -0.5
    assert rigor.parse_metric("nothing here", "loss") is None
    assert rigor.metric_spec("thr:max") == Metric("thr", True)
    assert rigor.metric_spec("loss") == Metric("loss", False)


if __name__ == "__main__":
    check_minimize()
    check_maximize()
    check_auto_margin_needs_noise()
    check_guardrails()
    check_old_ledger_best_without_guardrail_samples()
    check_sign_flip()
    check_paired_cancels_seed_variance()
    check_seed_reaches_command()
    check_parse()
    print("ok")
