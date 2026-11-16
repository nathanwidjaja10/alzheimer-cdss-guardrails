#!/usr/bin/env python3
"""Offline accuracy check of `classify_question()` (no Ollama needed).

Run it after changing the example bank (`_CATEGORY_EXAMPLES`), the keywords
(`_OUT_OF_SCOPE_KEYWORDS`) or the threshold (`_CLASSIFY_THRESHOLD`). The constants are imported
from `qa_engine.py`, so the report always reflects the production code.

Accuracy is reported separately for two question sets:
  1. results/qa_benchmark/qa_cases.jsonl  "dev": the benchmark questions used to diagnose errors.
  2. tools/qa_classifier_holdout.json      "generalization": differently worded held-out questions.

    python -m llm_xai.tools.tune_qa_classifier                 # accuracy at the current threshold
    python -m llm_xai.tools.tune_qa_classifier --grid          # plus a threshold sweep
    python -m llm_xai.tools.tune_qa_classifier --verbose       # plus the misclassified questions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llm_xai.backend import RESULTS_DIR  # noqa: E402
from llm_xai.backend.llm_engine import FactContext  # noqa: E402
from llm_xai.backend import qa_engine as qe  # noqa: E402

HOLDOUT_FP = Path(__file__).resolve().parent / "qa_classifier_holdout.json"
BENCH_FP = RESULTS_DIR / "qa_benchmark" / "qa_cases.jsonl"

# Placeholder fact context: `classify_question(q, fc)` does not use `fc`, but its signature needs one
_DUMMY_FC: FactContext | None = None


def _dummy_fc() -> FactContext:
    """Return a fact context taken from the first gold fact file (cached)."""
    global _DUMMY_FC
    if _DUMMY_FC is None:
        gold_dir = RESULTS_DIR / "gold_facts"
        fp = sorted(gold_dir.glob("*.json"))[0]
        _DUMMY_FC = FactContext(**json.loads(fp.read_text()))
    return _DUMMY_FC


def load_dev_cases() -> dict[str, str]:
    """Unique {question: expected_category} pairs of qa_cases.jsonl (classification ignores the session)."""
    if not BENCH_FP.exists():
        return {}
    uniq: dict[str, str] = {}
    for line in BENCH_FP.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        uniq[c["question"]] = c["expected_category"]
    return uniq


def load_holdout_cases() -> dict[str, str]:
    """Read the held-out {question: expected_category} pairs."""
    d = json.loads(HOLDOUT_FP.read_text())
    return {c["question"]: c["expected_category"] for c in d["cases"]}


def run(dataset: dict[str, str], threshold: float | None = None, verbose: bool = False) -> tuple[int, int]:
    """Return (n_correct, n_total) of `classify_question_debug` on a dataset.

    `threshold` temporarily overrides the production similarity threshold (restored afterwards).
    """
    fc = _dummy_fc()
    orig_threshold = qe._CLASSIFY_THRESHOLD
    if threshold is not None:
        qe._CLASSIFY_THRESHOLD = threshold
    try:
        n_correct = 0
        wrong: list[tuple[str, str, str, float | None, str | None]] = []
        for q, expected in dataset.items():
            d = qe.classify_question_debug(q, fc)
            pred = d["category"]
            if pred == expected:
                n_correct += 1
            else:
                wrong.append((expected, pred, q, d["similarity"], d["nearest_example"]))
        if verbose:
            for exp, pred, q, sim, nn in wrong:
                sim_s = f"{sim:.3f}" if sim is not None else "n/a (keyword)"
                print(f"    exp={exp:12s} got={pred:12s} sim={sim_s:14s} nn={nn!r}")
                print(f"        q={q!r}")
        return n_correct, len(dataset)
    finally:
        qe._CLASSIFY_THRESHOLD = orig_threshold


def main() -> int:
    """Report dev and generalization accuracy, optionally with a threshold sweep."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", action="store_true", help="sweep threshold 0.40-0.50 (step 0.01)")
    ap.add_argument("--verbose", action="store_true", help="show the wrong cases")
    args = ap.parse_args()

    dev = load_dev_cases()
    holdout = load_holdout_cases()
    if not dev:
        print(f"[tune_qa_classifier] WARNING: {BENCH_FP} is empty or missing -- "
              f"run build_qa_benchmark.py to obtain a dev score.")

    print(f"[tune_qa_classifier] current production threshold: {qe._CLASSIFY_THRESHOLD}")
    print(f"[tune_qa_classifier] bank: {sum(len(v) for v in qe._CATEGORY_EXAMPLES.values())} examples "
          f"across {len(qe._CATEGORY_EXAMPLES)} categories")
    print()

    if dev:
        nc, n = run(dev, verbose=args.verbose)
        print(f"dev (qa_cases.jsonl, {n} unique questions): {nc}/{n} = {nc/n:.1%}")
    nc, n = run(holdout, verbose=args.verbose)
    print(f"generalization (qa_classifier_holdout.json, {n} questions): {nc}/{n} = {nc/n:.1%}")

    if args.grid:
        print("\n[tune_qa_classifier] threshold sweep (example bank and keyword rules unchanged):")
        for t in [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48, 0.50]:
            dnc, dn = run(dev, threshold=t) if dev else (0, 0)
            hnc, hn = run(holdout, threshold=t)
            dev_s = f"{dnc}/{dn}={dnc/dn:.1%}" if dn else "n/a"
            print(f"  threshold={t:.2f}  dev={dev_s:16s}  generalization={hnc}/{hn}={hnc/hn:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
