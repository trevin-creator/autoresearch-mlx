"""Self-check for rigor.py's keep/discard rules, no real runs needed: uv run test_rigor.py"""

import tempfile
from pathlib import Path

import rigor


def make_study(maximize=False):
    tmp = Path(tempfile.mkdtemp())
    return rigor.Study(cmd=[], metric="m", files=[tmp / "exp.py"],
                       maximize=maximize, ledger=tmp / "ledger.jsonl")


def run(study, code, results, margin=0.03, seeds=3):
    study.files[0].write_text(code)
    it = iter(results)
    rigor.run_once = lambda _study: next(it)
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
    s = make_study(maximize=True)
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


def check_parse():
    out = "step 1\nloss: 3.2\nloss = 2.5e-1\nval_loss: 9\n"
    assert rigor.parse_metric(out, "loss") == 0.25  # last match wins
    assert rigor.parse_metric("acc: -.5\n", "acc") == -0.5
    assert rigor.parse_metric("nothing here", "loss") is None


if __name__ == "__main__":
    check_minimize()
    check_maximize()
    check_auto_margin_needs_noise()
    check_parse()
    print("ok")
