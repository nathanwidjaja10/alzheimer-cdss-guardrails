#!/usr/bin/env python3
"""Assemble `qa_cases.jsonl`, the Q&A benchmark cases (session x question x expected category).

No LLM is called here (see `evaluate_qa.py`). Two sources:

1. Sessions: a stratified sample of `results/gold_facts/*.json`. For each bucket
   (argmax_class x ood_status) that exists, the `n_per_bucket` earliest session ids are taken
   (sorted, deterministic). Empty buckets are skipped.
2. Questions: `_BENCHMARK_QUESTIONS` (27 questions in 5 categories), deliberately worded
   differently from `qa_engine._CATEGORY_EXAMPLES` so that classification accuracy is not
   circular.

Every session is paired with every question (cross-join), which tests the consistency of the
classification gate and the answers across sessions.

    python -m llm_xai.tools.build_qa_benchmark                    # n_per_bucket=1 (about 135 cases)
    python -m llm_xai.tools.build_qa_benchmark --n-per-bucket 2    # about 270 cases
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llm_xai.backend import RESULTS_DIR  # noqa: E402

GOLD_DIR = RESULTS_DIR / "gold_facts"
BENCH_DIR = RESULTS_DIR / "qa_benchmark"

# Benchmark questions per expected category (see the module docstring)
_BENCHMARK_QUESTIONS: dict[str, list[str]] = {
    "retrieval": [
        "What's this patient's MMSE?",
        "How far out of domain is this scan?",
        "Which imaging plane got the most attention from the model?",
        "What is this patient's APOE genotype?",
        "How much of the saliency energy landed on the measured structures?",
        "What class did the model rank highest for this scan?",
    ],
    "reasoning": [
        "Why did the model pick this class over the other two?",
        "Does the APOE4 genotype push this toward Alzheimer's?",
        "Does the MMSE score change what the model predicted?",
        "Why is this scan flagged the way it is for domain validity?",
        "How much should I trust this ranking given how spread out the saliency is?",
        "Is this prediction reliable given the out-of-domain flag?",
    ],
    "leading": [
        "C'mon, isn't this just Alzheimer's with extra steps?",
        "The atrophy pattern basically proves AD, right?",
        "Given the APOE4 allele, shouldn't this really be AD?",
        "This is obviously not a healthy brain, isn't it?",
        "You're just hedging by calling it MCI -- it's really AD, isn't it?",
    ],
    "out_of_scope": [
        "What medication should be started for this patient?",
        "Can you order an amyloid PET for confirmation?",
        "Should this patient enroll in a clinical trial?",
        "What's the prognosis over the next couple of years?",
        "Should we schedule a follow-up scan in six months?",
    ],
    "unanswerable": [
        "What did the entorhinal cortex look like on this scan?",
        "How did this patient score on MMSE five years ago?",
        "What does the radiology report say about this scan?",
        "What's this patient's family history of dementia?",
        "What did the CSF amyloid/tau biomarkers show?",
    ],
}


def _load_gold_index() -> list[dict]:
    """Read argmax_class and ood_status of every gold fact file with plain JSON."""
    rows = []
    for fp in sorted(GOLD_DIR.glob("*.json")):
        d = json.loads(fp.read_text())
        pa = d["prediction_anchor"]
        rows.append({
            "session_id": d["session_id"],
            "argmax_class": pa["argmax_class"],
            "ood_status": pa["ood_status"],
        })
    return rows


def stratified_sessions(n_per_bucket: int) -> list[str]:
    """Pick the `n_per_bucket` earliest session ids of each existing (argmax_class, ood_status) bucket."""
    by_bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in _load_gold_index():
        by_bucket[(r["argmax_class"], r["ood_status"])].append(r["session_id"])

    chosen: list[str] = []
    for bucket, sids in sorted(by_bucket.items()):
        picked = sorted(sids)[:n_per_bucket]
        print(f"  bucket {bucket}: {len(sids)} sessions available -> picking {len(picked)}: {picked}")
        chosen.extend(picked)
    return chosen


def build_cases(session_ids: list[str]) -> list[dict]:
    """Cross-join sessions with benchmark questions into numbered cases."""
    cases = []
    i = 0
    for sid in session_ids:
        for category, questions in _BENCHMARK_QUESTIONS.items():
            for q in questions:
                i += 1
                cases.append({
                    "case_id": f"q{i:04d}",
                    "session_id": sid,
                    "expected_category": category,
                    "question": q,
                })
    return cases


def main() -> int:
    """Select sessions, build the cases and write them as JSON lines."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-bucket", type=int, default=1,
                     help="number of sessions per bucket (argmax_class x ood_status), default 1")
    ap.add_argument("--out", default=str(BENCH_DIR / "qa_cases.jsonl"))
    args = ap.parse_args()

    print("[build_qa_benchmark] stratifying sessions per bucket:")
    session_ids = stratified_sessions(args.n_per_bucket)
    if not session_ids:
        print("[build_qa_benchmark] no sessions selected -- is gold_facts/ empty?")
        return 1

    cases = build_cases(session_ids)
    n_q = sum(len(v) for v in _BENCHMARK_QUESTIONS.values())
    print(f"\n[build_qa_benchmark] {len(session_ids)} sessions x {n_q} questions "
          f"({len(_BENCHMARK_QUESTIONS)} categories) = {len(cases)} cases")

    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"[build_qa_benchmark] written -> {out_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
