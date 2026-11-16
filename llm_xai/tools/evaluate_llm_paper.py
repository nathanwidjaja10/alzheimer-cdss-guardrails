#!/usr/bin/env python3
"""Narrative factuality benchmark for the paper.

For each gold fact x model x variant (a, b, c): generate_from_facts -> run_dag -> record the
numeric consistency rate (NCR), violation types, latency and call count. All raw results are then
aggregated into tables T1-T4 (CSV), a faithfulness-by-field table, McNemar tests and
`run_manifest.json`.

Use `--one-model-at-a-time` (pull, sweep, `ollama rm`) to limit disk use and `--resume` to skip
existing combinations. A full sweep takes hours and is best run in stages.

    python -m llm_xai.tools.evaluate_llm_paper --variant c --models biomistral --limit 10
    python -m llm_xai.tools.evaluate_llm_paper --variant a,b,c --models biomistral      # ablation (T2)
    python -m llm_xai.tools.evaluate_llm_paper --variant c --one-model-at-a-time        # all models (T1)

Cloud models are rejected unless `--allow-cloud` is given and `configs/ethics_clearance.json`
exists; by default only `kind: ollama` models run.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llm_xai.backend import RESULTS_DIR  # noqa: E402
from llm_xai.backend.guardrails import (  # noqa: E402
    _NLI_ENTAILMENT_THRESHOLD,
    narrative_faithfulness_per_field,
)
from llm_xai.backend.llm_engine import FactContext, NarrativeOut, run_manifest  # noqa: E402
from llm_xai.backend.pipeline import generate_from_facts, load_models_cfg  # noqa: E402
from llm_xai.tools.stats_utils import mcnemar_exact_p, mcnemar_table, wilson_ci  # noqa: E402

GOLD_DIR = RESULTS_DIR / "gold_facts"
RAW_DIR = RESULTS_DIR / "eval_raw"       # one JSON file per (session, model, variant)


def load_gold(limit: int = 0) -> list[FactContext]:
    """Load the gold fact contexts (first `limit` when given)."""
    files = sorted(GOLD_DIR.glob("*.json"))
    if limit:
        files = files[:limit]
    return [FactContext(**json.loads(f.read_text())) for f in files]


@functools.lru_cache(maxsize=None)
def _gold_fact_by_session(session_id: str) -> FactContext | None:
    """Gold fact context of one session (cached; a session is reused across model/variant rows)."""
    fp = GOLD_DIR / f"{session_id}.json"
    if not fp.exists():
        return None
    return FactContext(**json.loads(fp.read_text()))


def _row_faithfulness(row: dict) -> dict[str, float] | None:
    """Per-field NLI faithfulness {field_path: score} of the narrative stored in a raw row.

    Uses the stored narrative only (no LLM call); returns None when the row has no narrative or no
    gold fact.
    """
    narr_dict = row.get("narrative")
    if not narr_dict:
        return None
    facts = _gold_fact_by_session(row["session_id"])
    if facts is None:
        return None
    return narrative_faithfulness_per_field(NarrativeOut(**narr_dict), facts)


def load_all_raw() -> list[dict]:
    """Read every raw result file, so the tables always cover all sweeps accumulated so far."""
    rows = []
    for fp in sorted(RAW_DIR.glob("*__*__*.json")):
        try:
            rows.append(json.loads(fp.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {fp.name}: failed to read ({e})")
    return rows


def ollama_rm(model_tag: str) -> None:
    """Remove a model from the local Ollama store (no-op when ollama is not installed)."""
    try:
        subprocess.run(["ollama", "rm", model_tag], check=False, capture_output=True)
    except FileNotFoundError:
        pass


def eval_one(fc: FactContext, model_key: str, variant: str, allow_cloud: bool) -> dict:
    """Generate and audit one narrative; return the raw result row."""
    t0 = time.perf_counter()
    res = generate_from_facts(fc, model_key, variant, allow_cloud=allow_cloud)
    wall = time.perf_counter() - t0
    g = res.get("guardrail")  # None when the output could not be parsed
    meta = res.get("llm_meta", {}) or {}
    tok = meta.get("eval_count") or 0
    dur_ns = meta.get("eval_duration_ns") or 0
    parse_error = res.get("parse_error")
    return {
        "session_id": fc.session_id,
        "model": model_key,
        "variant": variant,
        "parse_error": bool(parse_error),
        "passed": bool(g["passed"]) if g else False,
        "source": g["source"] if g else "parse_error",
        "ncr": g["ncr"] if g else 0.0,
        "n_violations": sum(len(a["violations"]) for a in g["audits"]) if g else None,
        "violation_types": ";".join(g["violation_types"]) if g else "parse_error",
        "llm_calls": res.get("llm_calls", 1),
        "wall_s": round(wall, 2),
        "tokens": tok,
        "ms_per_token": round((dur_ns / 1e6 / tok), 2) if tok else None,
        "aseg_available": fc.spatial_xai_anchor.aseg_available,
        "ood_status": fc.prediction_anchor.ood_status,
        "argmax_class": fc.prediction_anchor.argmax_class,
        "narrative": res.get("narrative"),
        "guardrail": g,
    }


def aggregate(rows: list[dict], compute_faithfulness: bool = True) -> dict[str, list[dict]]:
    """Aggregate raw rows per (model, variant) into the result tables T1-T4 and faithfulness by field."""
    by_mv: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_mv[(r["model"], r["variant"])].append(r)

    t1, t2, t3, t4 = [], [], [], []
    faith_by_field: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    # (model, variant, field_path) -> faithfulness scores, for T1_faithfulness_by_field

    for (model, variant), rs in sorted(by_mv.items()):
        n = len(rs)
        n_fail = sum(0 if r["passed"] else 1 for r in rs)
        hr = n_fail / n * 100 if n else 0.0
        hr_lo, hr_hi = wilson_ci(n_fail, n)
        ncr = sum(r["ncr"] for r in rs) / n if n else 0.0
        # Per-field NLI faithfulness, computed from the stored narratives (no LLM call) and pooled over
        # (narrative, field) pairs, so the pass percentage is over pairs, not narratives.
        # --skip-faithfulness omits this block.
        faith_scores: list[float] = []
        if compute_faithfulness:
            for r in rs:
                field_scores = _row_faithfulness(r)
                if field_scores is None:
                    continue
                for field, s in field_scores.items():
                    faith_scores.append(s)
                    faith_by_field[(model, variant, field)].append(s)
        n_faith_pass = sum(1 for s in faith_scores if s >= _NLI_ENTAILMENT_THRESHOLD)
        faith_lo, faith_hi = wilson_ci(n_faith_pass, len(faith_scores)) if faith_scores else (None, None)
        # Pooled NCR: numeric-auditor valid/checked summed over sessions, so every numeric claim has
        # equal weight. Sessions without a guardrail report (parse errors) are excluded; they are
        # counted in parse_error_pct.
        checked_total = valid_total = 0
        for r in rs:
            g = r.get("guardrail")
            if not g:
                continue
            for a in g.get("audits", []):
                if a["name"] == "numeric":
                    checked_total += a["checked"]
                    valid_total += a["valid"]
        ncr_lo, ncr_hi = wilson_ci(valid_total, checked_total) if checked_total else (None, None)
        fb = sum(1 for r in rs if r["source"] == "deterministic_fallback") / n * 100 if n else 0.0
        pe = sum(1 for r in rs if r.get("parse_error")) / n * 100 if n else 0.0
        regen = sum(1 for r in rs if r["llm_calls"] > 1) / n * 100 if n else 0.0
        lat = sorted(r["wall_s"] for r in rs)
        mpt = sorted(r["ms_per_token"] for r in rs if r["ms_per_token"])
        row = {
            "model": model, "variant": variant, "n": n,
            "HR_pct": round(hr, 2),
            "HR_wilson_lo_pct": round(hr_lo * 100, 2), "HR_wilson_hi_pct": round(hr_hi * 100, 2),
            "NCR_mean": round(ncr, 4),
            "NCR_pooled": round(valid_total / checked_total, 4) if checked_total else None,
            "NCR_pooled_n_claims": checked_total,
            "NCR_wilson_lo": round(ncr_lo, 4) if ncr_lo is not None else None,
            "NCR_wilson_hi": round(ncr_hi, 4) if ncr_hi is not None else None,
            "Faithfulness_NLI_mean": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
            "Faithfulness_n": len(faith_scores),
            "Faithfulness_pass_pct": round(n_faith_pass / len(faith_scores) * 100, 2) if faith_scores else None,
            "Faithfulness_wilson_lo_pct": round(faith_lo * 100, 2) if faith_lo is not None else None,
            "Faithfulness_wilson_hi_pct": round(faith_hi * 100, 2) if faith_hi is not None else None,
            "parse_error_pct": round(pe, 2),
            "fallback_pct": round(fb, 2), "regenerated_pct": round(regen, 2),
            "wall_s_median": round(statistics.median(lat), 2) if lat else None,
            "ms_per_token_median": round(statistics.median(mpt), 2) if mpt else None,
        }
        if variant == "c":
            t1.append(row)          # T1: cross-model comparison (variant c)
        t2.append(row)             # T2: ablation over all variants

        # T3: violation types
        vc: Counter = Counter()
        for r in rs:
            vc.update(filter(None, r["violation_types"].split(";")))
        for vt, cnt in vc.most_common():
            t3.append({"model": model, "variant": variant, "violation_type": vt, "count": cnt})

        # T4: sessions outside the validated domain
        ood_rs = [r for r in rs if r["ood_status"] == "outside_validated_domain"]
        if ood_rs:
            t4.append({
                "model": model, "variant": variant, "n_outside_domain": len(ood_rs),
                "HR_pct": round(sum(0 if r["passed"] else 1 for r in ood_rs) / len(ood_rs) * 100, 2),
            })

    # Faithfulness by field: shows which fields are weak behind the pooled average
    t1_field = []
    for (model, variant, field), scores in sorted(faith_by_field.items()):
        n_f = len(scores)
        n_pass_f = sum(1 for s in scores if s >= _NLI_ENTAILMENT_THRESHOLD)
        lo_f, hi_f = wilson_ci(n_pass_f, n_f) if n_f else (None, None)
        t1_field.append({
            "model": model, "variant": variant, "field": field, "n": n_f,
            "mean": round(sum(scores) / n_f, 4) if n_f else None,
            "pass_pct": round(n_pass_f / n_f * 100, 2) if n_f else None,
            "pass_wilson_lo_pct": round(lo_f * 100, 2) if lo_f is not None else None,
            "pass_wilson_hi_pct": round(hi_f * 100, 2) if hi_f is not None else None,
        })

    return {"T1_models_variantC": t1, "T2_ablation": t2, "T3_violation_breakdown": t3,
            "T4_out_of_domain": t4, "T1_faithfulness_by_field": t1_field}


def compute_mcnemar_table(rows: list[dict]) -> list[dict]:
    """Paired exact McNemar tests over the same sessions.

    Two kinds of contrast: (1) variants of the same model (a vs b, a vs c, b vs c); (2) models
    on the same variant. Only pairs with at least one shared session are reported.
    """
    by_mv: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for r in rows:
        by_mv[(r["model"], r["variant"])][r["session_id"]] = bool(r["passed"])

    out: list[dict] = []

    def add_row(contrast: str, label_x: str, label_y: str, key_x: tuple[str, str], key_y: tuple[str, str]) -> None:
        if key_x not in by_mv or key_y not in by_mv:
            return
        b, c, n_pairs = mcnemar_table(by_mv[key_x], by_mv[key_y])
        if n_pairs == 0:
            return
        p = mcnemar_exact_p(b, c)
        out.append({
            "contrast": contrast, "x": label_x, "y": label_y, "n_pairs": n_pairs,
            "pass_x_fail_y": b, "fail_x_pass_y": c, "n_discordant": b + c,
            "mcnemar_p": round(p, 6), "significant_p05": p < 0.05,
        })

    models = sorted({m for m, _ in by_mv})
    variants_by_model: dict[str, list[str]] = defaultdict(list)
    for m, v in by_mv:
        variants_by_model[m].append(v)

    # Variant ablation within a model
    for m in models:
        vs = sorted(variants_by_model[m])
        for i, vx in enumerate(vs):
            for vy in vs[i + 1:]:
                add_row("ablation_variant", f"{m}/{vx}", f"{m}/{vy}", (m, vx), (m, vy))

    # Different models on the same variant
    for variant in ("a", "b", "c"):
        ms = [m for m in models if (m, variant) in by_mv]
        for i, mx in enumerate(ms):
            for my in ms[i + 1:]:
                add_row(f"cross_model_variant_{variant}", f"{mx}/{variant}", f"{my}/{variant}",
                        (mx, variant), (my, variant))

    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as CSV with LF line endings (empty file when there are no rows)."""
    if not rows:
        path.write_text("")
        return
    # Explicit LF line terminator: the csv default (CRLF) would differ from the repository's normalised line endings
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    """Run the sweep (unless --stats-only), then aggregate all raw results and write the tables and manifest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="biomistral", help="comma-separated list")
    ap.add_argument("--variant", "--variants", dest="variants", default="c", help="comma-separated a|b|c")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--one-model-at-a-time", action="store_true",
                    help="run models one at a time; `ollama rm` after each finishes (saves disk)")
    ap.add_argument("--allow-cloud", action="store_true", help="allow cloud-kind models (needs ethics_clearance.json)")
    ap.add_argument("--stats-only", action="store_true",
                    help="skip generation (Ollama not needed); re-aggregate the existing eval_raw/ into T1-T4 and mcnemar_tests.csv")
    ap.add_argument("--skip-faithfulness", action="store_true",
                    help="skip the NLI faithfulness metric during aggregation (faster; T1/T2 lose the Faithfulness_* columns)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    n_sessions = 0
    if not args.stats_only:
        gold = load_gold(args.limit)
        if not gold:
            print(f"[evaluate] no gold facts in {GOLD_DIR}. Run build_gold_facts first.")
            return 1
        n_sessions = len(gold)
        print(f"[evaluate] {len(gold)} sessions x {len(models)} models x {len(variants)} variants")

        cfg = load_models_cfg()
        rows: list[dict] = []
        for model_key in models:
            model_tag = cfg.get(model_key, {}).get("model", model_key)
            for variant in variants:
                for i, fc in enumerate(gold, 1):
                    raw_fp = RAW_DIR / f"{fc.session_id}__{model_key}__{variant}.json"
                    if args.resume and raw_fp.exists():
                        rows.append(json.loads(raw_fp.read_text()))
                        continue
                    try:
                        r = eval_one(fc, model_key, variant, args.allow_cloud)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {fc.session_id} [{model_key}/{variant}]  ERROR {e}")
                        continue
                    raw_fp.write_text(json.dumps(r, indent=2, default=str))
                    rows.append(r)
                    if i % 10 == 0 or i == len(gold):
                        print(f"  {model_key}/{variant}: {i}/{len(gold)}  (HR so far "
                              f"{sum(0 if x['passed'] else 1 for x in rows if x['model']==model_key and x['variant']==variant)})")
            if args.one_model_at_a_time and cfg.get(model_key, {}).get("kind", "ollama") == "ollama":
                print(f"  [disk] ollama rm {model_tag}")
                ollama_rm(model_tag)
    else:
        print("[evaluate] --stats-only: skipping generation, re-aggregating existing eval_raw/")

    all_rows = load_all_raw()
    tables = aggregate(all_rows, compute_faithfulness=not args.skip_faithfulness)
    for name, tbl in tables.items():
        write_csv(RESULTS_DIR / f"{name}.csv", tbl)
        print(f"[evaluate] {name}: {len(tbl)} rows -> {RESULTS_DIR / (name + '.csv')}")

    mcnemar_rows = compute_mcnemar_table(all_rows)
    write_csv(RESULTS_DIR / "mcnemar_tests.csv", mcnemar_rows)
    print(f"[evaluate] mcnemar_tests: {len(mcnemar_rows)} rows -> {RESULTS_DIR / 'mcnemar_tests.csv'}")

    seen_mv = sorted({(r["model"], r["variant"]) for r in all_rows})
    manifest = run_manifest()
    manifest.update({"n_sessions": n_sessions or len({r["session_id"] for r in all_rows}),
                     "models": models, "variants": variants,
                     "n_raw_total": len(all_rows), "model_variant_combos_in_csv": seen_mv,
                     "stats_only_run": args.stats_only,
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if not args.stats_only:
        (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[evaluate] run_manifest.json written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
