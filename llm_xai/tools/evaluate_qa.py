#!/usr/bin/env python3
"""Q&A benchmark: run `qa_cases.jsonl` through `answer_followup()` and write the T5 tables.

Companion of `evaluate_llm_paper.py` (one-shot narratives). For each case (session_id, question,
expected_category):
  start_qa_session (the base narrative comes from the narrative cache when available)
  -> qa.turns = [] (every case is evaluated as an independent single turn)
  -> answer_followup -> record the predicted category, guardrail report, source and latency

Cost: each case makes one or two real Ollama calls (about 15-30 s each with llama31_8b), so the
default 270 cases (`build_qa_benchmark.py --n-per-bucket 2`) take roughly 1-2 hours. Run
`--limit N` first as a smoke test.

    python -m llm_xai.tools.evaluate_qa --limit 5                # smoke test
    python -m llm_xai.tools.evaluate_qa --resume                 # continue an interrupted run
    python -m llm_xai.tools.evaluate_qa --stats-only              # re-aggregate only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llm_xai.backend import RESULTS_DIR  # noqa: E402
from llm_xai.backend.pipeline import start_qa_session  # noqa: E402
from llm_xai.backend.qa_engine import answer_followup  # noqa: E402
from llm_xai.tools.stats_utils import wilson_ci  # noqa: E402

BENCH_DIR = RESULTS_DIR / "qa_benchmark"
CASES_FP = BENCH_DIR / "qa_cases.jsonl"
RAW_DIR = RESULTS_DIR / "eval_qa_raw"


def load_cases(limit: int = 0) -> list[dict]:
    """Read the benchmark cases (first `limit` when given); empty when the file is missing."""
    if not CASES_FP.exists():
        return []
    cases = [json.loads(line) for line in CASES_FP.read_text().splitlines() if line.strip()]
    if limit:
        cases = cases[:limit]
    return cases


def load_all_raw() -> list[dict]:
    """Read every raw result file, so T5 always covers all runs accumulated so far."""
    rows = []
    for fp in sorted(RAW_DIR.glob("*.json")):
        try:
            rows.append(json.loads(fp.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {fp.name}: failed to read ({e})")
    return rows


def eval_one(case: dict, model_key: str, qa_cache: dict) -> dict:
    """Answer one benchmark case and return the raw result row."""
    sid = case["session_id"]
    if sid not in qa_cache:
        qa_cache[sid] = start_qa_session(sid, model_key=model_key, variant="c", use_cache=True)
    qa = qa_cache[sid]
    qa.turns = []  # each case is an independent single turn

    t0 = time.perf_counter()
    turn = answer_followup(qa, case["question"], model_key=model_key)
    wall = time.perf_counter() - t0

    g = turn.guardrail or {}
    expected = case["expected_category"]
    is_refusal_expected = expected in ("out_of_scope", "unanswerable")
    actually_refused = turn.source == "deterministic_refusal"
    a_out = turn.answer_out

    return {
        "case_id": case["case_id"], "session_id": sid, "question": case["question"],
        "expected_category": expected, "predicted_category": turn.category,
        "category_correct": turn.category == expected,
        "source": turn.source,
        "guardrail_passed": bool(g.get("passed")),
        "guardrail": g,
        "llm_calls": turn.llm_calls, "wall_s": round(wall, 2),
        "is_refusal_expected": is_refusal_expected, "actually_refused": actually_refused,
        # Rejected first draft and correction, when the answer was regenerated or refused
        "attempt1": turn.attempt1, "correction_used": turn.correction_used,
        "attempt2": turn.attempt2,
        "answerable": a_out.answerable if a_out else None,
        "cited_fields": a_out.cited_fields if a_out else [],
        "answer": a_out.answer if a_out else None,
    }


def _audit_failed(g: dict, name: str) -> bool:
    """True when the named auditor failed in a guardrail report."""
    return any(a["name"] == name and not a["passed"] for a in g.get("audits", []))


def aggregate(rows: list[dict]) -> dict[str, list[dict]]:
    """Compute the overall and per-category T5 tables (accuracy, hallucination rate, refusal precision and recall)."""
    if not rows:
        return {"T5_qa_overall": [], "T5_qa_per_category": []}

    n = len(rows)
    n_correct = sum(1 for r in rows if r["category_correct"])
    acc_lo, acc_hi = wilson_ci(n_correct, n)

    n_fail = sum(1 for r in rows if not r["guardrail_passed"])
    hr_lo, hr_hi = wilson_ci(n_fail, n)

    n_grd_fail = sum(1 for r in rows if _audit_failed(r["guardrail"], "grounding"))
    n_con_fail = sum(1 for r in rows if _audit_failed(r["guardrail"], "contradiction"))

    # Refusal precision and recall (a refusal is expected for out-of-scope and unanswerable questions)
    tp = sum(1 for r in rows if r["is_refusal_expected"] and r["actually_refused"])
    fp = sum(1 for r in rows if not r["is_refusal_expected"] and r["actually_refused"])
    fn = sum(1 for r in rows if r["is_refusal_expected"] and not r["actually_refused"])
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None

    lat = sorted(r["wall_s"] for r in rows)

    overall = [{
        "n": n,
        "category_accuracy_pct": round(n_correct / n * 100, 2),
        "category_accuracy_wilson_lo_pct": round(acc_lo * 100, 2),
        "category_accuracy_wilson_hi_pct": round(acc_hi * 100, 2),
        "HR_per_answer_pct": round(n_fail / n * 100, 2),
        "HR_wilson_lo_pct": round(hr_lo * 100, 2),
        "HR_wilson_hi_pct": round(hr_hi * 100, 2),
        "grounding_fail_pct": round(n_grd_fail / n * 100, 2),
        "contradiction_fail_pct": round(n_con_fail / n * 100, 2),
        "refusal_precision": precision, "refusal_recall": recall,
        "refusal_tp": tp, "refusal_fp": fp, "refusal_fn": fn,
        "regenerated_pct": round(sum(1 for r in rows if r["llm_calls"] > 1) / n * 100, 2),
        "wall_s_median": round(statistics.median(lat), 2) if lat else None,
    }]

    # Per expected category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["expected_category"]].append(r)

    per_category = []
    for cat, rs in sorted(by_cat.items()):
        n_c = len(rs)
        n_fail_c = sum(1 for r in rs if not r["guardrail_passed"])
        n_correct_c = sum(1 for r in rs if r["category_correct"])
        src = Counter(r["source"] for r in rs)
        per_category.append({
            "category": cat, "n": n_c,
            "category_accuracy_pct": round(n_correct_c / n_c * 100, 2),
            "HR_per_answer_pct": round(n_fail_c / n_c * 100, 2),
            "source_llm": src.get("llm", 0),
            "source_llm_regenerated": src.get("llm_regenerated", 0),
            "source_deterministic_refusal": src.get("deterministic_refusal", 0),
        })

    return {"T5_qa_overall": overall, "T5_qa_per_category": per_category}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as CSV with LF line endings (empty file when there are no rows)."""
    import csv
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    """Run the benchmark (unless --stats-only), then aggregate all raw results into the T5 tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama31_8b",
                     help="model key from llm_models.yaml")
    ap.add_argument("--limit", type=int, default=0, help="limit the number of cases (smoke test)")
    ap.add_argument("--resume", action="store_true", help="skip cases whose raw file already exists")
    ap.add_argument("--stats-only", action="store_true",
                     help="skip evaluation (no need for Ollama to be running); just re-aggregate eval_qa_raw/")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.stats_only:
        cases = load_cases(args.limit)
        if not cases:
            print(f"[evaluate_qa] no cases in {CASES_FP}. Run build_qa_benchmark.py first.")
            return 1
        print(f"[evaluate_qa] {len(cases)} cases, model={args.model}")

        qa_cache: dict[str, object] = {}
        for i, case in enumerate(cases, 1):
            raw_fp = RAW_DIR / f"{case['case_id']}.json"
            if args.resume and raw_fp.exists():
                continue
            try:
                r = eval_one(case, args.model, qa_cache)
            except Exception as e:  # noqa: BLE001
                print(f"  {case['case_id']} [{case['session_id']}]  ERROR {e}")
                continue
            raw_fp.write_text(json.dumps(r, indent=2, default=str))
            if i % 10 == 0 or i == len(cases):
                n_fail_so_far = sum(1 for fp in RAW_DIR.glob("*.json")
                                     if not json.loads(fp.read_text()).get("guardrail_passed"))
                print(f"  {i}/{len(cases)}  (HR so far, across all raw files: {n_fail_so_far})")
    else:
        print("[evaluate_qa] --stats-only: skipping evaluation, re-aggregating existing eval_qa_raw/")

    all_rows = load_all_raw()
    tables = aggregate(all_rows)
    for name, tbl in tables.items():
        write_csv(RESULTS_DIR / f"{name}.csv", tbl)
        print(f"[evaluate_qa] {name}: {len(tbl)} rows -> {RESULTS_DIR / (name + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
