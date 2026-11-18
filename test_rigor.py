"""Self-check for rigor.py's keep/discard rules, no training needed: uv run test_rigor.py"""

import tempfile
from pathlib import Path

import rigor


def run(results, seeds=3):
    it = iter(results)
    rigor.run_once = lambda: next(it)
    rigor.score("t", seeds, 0.95, 0.03)
    return rigor.load_ledger()[-1]


def main():
    tmp = Path(tempfile.mkdtemp())
    rigor.TRAIN, rigor.LEDGER = tmp / "train.py", tmp / "ledger.jsonl"

    rigor.TRAIN.write_text("a")
    assert run([1.50, 1.50, 1.50])["status"] == "keep"  # baseline

    # first run slightly worse (within noise) must NOT be discarded early
    rigor.TRAIN.write_text("b")
    e = run([1.51, 1.40, 1.40])
    assert len(e["samples"]) == 3, e

    # first run worse by more than the margin is discarded after one run
    rigor.TRAIN.write_text("c")
    e = run([1.60])
    assert e["status"] == "discard" and len(e["samples"]) == 1, e

    # a crash doesn't block retrying the same train.py
    rigor.TRAIN.write_text("d")
    assert run([None])["status"] == "crash"
    n = len(rigor.load_ledger())
    run([1.45, 1.45, 1.45])
    assert len(rigor.load_ledger()) == n + 1

    # but a scored config is still never re-run
    run([1.0, 1.0, 1.0])
    assert len(rigor.load_ledger()) == n + 1
    print("ok")


if __name__ == "__main__":
    main()
