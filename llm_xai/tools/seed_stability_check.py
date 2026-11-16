#!/usr/bin/env python3
"""Seed stability of one-shot narratives: how much do generations differ across sampling seeds?

For every session x seed, `generate_from_facts(fc, "llama31_8b", "c", seed_override=seed)` is run
and the differential conclusion, source, guardrail result and narrative text are recorded. The
narrative cache is not used, so each seed is a separate generation.

Sessions are the unique session ids of `results/qa_benchmark/qa_cases.jsonl` (25 sessions);
seeds are 42 (production default), 7 and 123. Outputs go to `results/seed_stability_raw/` and
`T6_seed_stability*.csv`; `eval_raw/` and `eval_qa_raw/` are not touched.

Cost: 25 sessions x 3 seeds = 75 real Ollama generations (about 30-40 minutes).

    python -m llm_xai.tools.seed_stability_check                 # full run
    python -m llm_xai.tools.seed_stability_check --limit 3        # smoke test
    python -m llm_xai.tools.seed_stability_check --resume         # continue an interrupted run
    python -m llm_xai.tools.seed_stability_check --stats-only     # re-aggregate only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llm_xai.backend import RESULTS_DIR  # noqa: E402
from llm_xai.backend.llm_engine import FactContext  # noqa: E402
from llm_xai.backend.pipeline import generate_from_facts  # noqa: E402
from llm_xai.tools.stats_utils import wilson_ci  # noqa: E402

GOLD_DIR = RESULTS_DIR / "gold_facts"
QA_CASES_FP = RESULTS_DIR / "qa_benchmark" / "qa_cases.jsonl"
RAW_DIR = RESULTS_DIR / "seed_stability_raw"
SEEDS = [42, 7, 123]
MODEL_KEY = "llama31_8b"
VARIANT = "c"


def load_sessions(limit: int = 0) -> list[str]:
    """Unique session ids of the Q&A benchmark cases, sorted (first `limit` when given)."""
    sids: list[str] = []
    seen: set[str] = set()
    with open(QA_CASES_FP, "r", encoding="utf-8") as f:
        for line in f:
            sid = json.loads(line)["session_id"]
            if sid not in seen:
                seen.add(sid)
                sids.append(sid)
    sids.sort()
    if limit:
        sids = sids[:limit]
    return sids


def eval_one(fc: FactContext, seed: int) -> dict:
    """Generate and audit one narrative for a seed; return the raw result row."""
    t0 = time.perf_counter()
    res = generate_from_facts(fc, MODEL_KEY, VARIANT, seed_override=seed)
    wall = time.perf_counter() - t0
    g = res.get("guardrail")
    narr = res.get("narrative")
    return {
        "session_id": fc.session_id, "seed": seed,
        "differential_conclusion": narr.get("differential_conclusion") if narr else None,
        "source": g["source"] if g else "parse_error",
        "passed": bool(g["passed"]) if g else False,
        "ncr": g["ncr"] if g else 0.0,
        "llm_calls": res.get("llm_calls", 1),
        "wall_s": round(wall, 2),
        "narrative": narr,
        "guardrail": g,
    }


def load_all_raw() -> list[dict]:
    """Read every raw result file."""
    rows = []
    for fp in sorted(RAW_DIR.glob("*.json")):
        try:
            rows.append(json.loads(fp.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {fp.name}: failed to read ({e})")
    return rows


def _narrative_text(narr: dict | None) -> str:
    """Join the four free-text sections (not the categorical conclusion) for similarity comparison."""
    if not narr:
        return ""
    parts = [narr.get(k, "") for k in
             ("visual_findings", "multimodal_correlation", "differential_justification", "uncertainty_note")]
    return " ".join(p for p in parts if p)


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return the per-session and overall seed-stability tables.

    Measures stability of the differential conclusion, mean pairwise cosine similarity of the narrative
    embeddings across seeds, hallucination rate and pooled numeric consistency.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Group rows by session and seed
    by_session: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_session.setdefault(r["session_id"], {})[r["seed"]] = r

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Accumulators for the overall table
    per_session = []
    all_concl_stable = 0
    all_source_counts: dict[str, int] = {}
    all_hr_fail = 0
    all_ncr_checked = all_ncr_valid = 0
    all_sim: list[float] = []
    n_sessions = 0

    for sid in sorted(by_session):
        by_seed = by_session[sid]
        if set(by_seed) != set(SEEDS):
            print(f"  [skip aggregation] {sid}: incomplete seeds ({sorted(by_seed)}), skipping this session")
            continue
        n_sessions += 1
        # Conclusion agreement and narrative similarity across seeds
        concls = {seed: by_seed[seed]["differential_conclusion"] for seed in SEEDS}
        stable = len(set(concls.values())) == 1
        if stable:
            all_concl_stable += 1
        texts = [_narrative_text(by_seed[seed]["narrative"]) for seed in SEEDS]
        embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        pair_sims = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                pair_sims.append(float(np.dot(embs[i], embs[j])))
        mean_sim = sum(pair_sims) / len(pair_sims) if pair_sims else None
        if mean_sim is not None:
            all_sim.append(mean_sim)
        for seed in SEEDS:
            r = by_seed[seed]
            all_source_counts[r["source"]] = all_source_counts.get(r["source"], 0) + 1
            if not r["passed"]:
                all_hr_fail += 1
            g = r.get("guardrail") or {}
            for a in g.get("audits", []):
                if a["name"] == "numeric":
                    all_ncr_checked += a["checked"]
                    all_ncr_valid += a["valid"]
        per_session.append({
            "session_id": sid,
            "differential_conclusion_stable": stable,
            "differential_conclusion_by_seed": ";".join(f"{s}={concls[s]}" for s in SEEDS),
            "mean_pairwise_cosine_sim": round(mean_sim, 4) if mean_sim is not None else None,
            "source_by_seed": ";".join(f"{s}={by_seed[s]['source']}" for s in SEEDS),
            "any_hr": any(not by_seed[s]["passed"] for s in SEEDS),
        })

    n_gen = n_sessions * len(SEEDS)
    hr_lo, hr_hi = wilson_ci(all_hr_fail, n_gen) if n_gen else (None, None)
    concl_lo, concl_hi = wilson_ci(all_concl_stable, n_sessions) if n_sessions else (None, None)
    overall = [{
        "n_sessions": n_sessions, "n_seeds": len(SEEDS), "n_generations": n_gen,
        "differential_conclusion_stable_pct": round(all_concl_stable / n_sessions * 100, 2) if n_sessions else None,
        "differential_conclusion_stable_wilson_lo_pct": round(concl_lo * 100, 2) if concl_lo is not None else None,
        "differential_conclusion_stable_wilson_hi_pct": round(concl_hi * 100, 2) if concl_hi is not None else None,
        "mean_pairwise_cosine_sim": round(sum(all_sim) / len(all_sim), 4) if all_sim else None,
        "HR_per_generation_pct": round(all_hr_fail / n_gen * 100, 2) if n_gen else None,
        "HR_wilson_lo_pct": round(hr_lo * 100, 2) if hr_lo is not None else None,
        "HR_wilson_hi_pct": round(hr_hi * 100, 2) if hr_hi is not None else None,
        "NCR_pooled": round(all_ncr_valid / all_ncr_checked, 4) if all_ncr_checked else None,
        "NCR_pooled_n_claims": all_ncr_checked,
        "source_counts": json.dumps(all_source_counts),
    }]
    return per_session, overall


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as a CSV file."""
    import csv
    if not rows:
        print(f"  [skip] {path.name}: no rows")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[seed_stability_check] {path.name}: {len(rows)} rows -> {path}")


def main() -> int:
    """Run the missing generations (unless --stats-only), then aggregate and write the tables."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit the number of sessions (smoke test)")
    ap.add_argument("--resume", action="store_true", help="skip (session,seed) pairs whose file already exists")
    ap.add_argument("--stats-only", action="store_true", help="re-aggregate only (Ollama not needed)")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.stats_only:
        sids = load_sessions(args.limit)
        print(f"[seed_stability_check] {len(sids)} sessions x {len(SEEDS)} seeds = {len(sids) * len(SEEDS)} generations, model={MODEL_KEY}")
        n_done = n_skip = 0
        for sid in sids:
            fp_fc = GOLD_DIR / f"{sid}.json"
            if not fp_fc.exists():
                print(f"  [skip] {sid}: gold facts missing")
                continue
            fc = FactContext(**json.loads(fp_fc.read_text()))
            for seed in SEEDS:
                out_fp = RAW_DIR / f"{sid}__seed{seed}.json"
                if args.resume and out_fp.exists():
                    n_skip += 1
                    continue
                row = eval_one(fc, seed)
                out_fp.write_text(json.dumps(row, indent=2))
                n_done += 1
                if n_done % 10 == 0:
                    print(f"  {n_done} new generations done ({n_skip} skipped/already present)")
        print(f"[seed_stability_check] done: {n_done} new, {n_skip} skipped")

    rows = load_all_raw()
    per_session, overall = aggregate(rows)
    write_csv(RESULTS_DIR / "T6_seed_stability_per_session.csv", per_session)
    write_csv(RESULTS_DIR / "T6_seed_stability.csv", overall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
