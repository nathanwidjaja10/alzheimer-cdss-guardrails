"""Summarize the blind human audit from audit_log.jsonl.

May be run while the audit is in progress (progress counts) or at the end (statistics and the
summary table).

    python3 audit/summarize_audit.py                 # progress and statistics
    python3 audit/summarize_audit.py --progress-only  # progress counts only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
LOG_PATH = AUDIT_DIR / "results" / "audit_log.jsonl"
MANIFEST_PATH = AUDIT_DIR / "cases_manifest.json"
OUT_CSV = AUDIT_DIR.parent / "results" / "T7_human_audit_summary.csv"


def wilson_upper(successes: int, n: int, z: float = 1.96) -> float:
    """Upper bound of the Wilson score interval for a proportion."""
    if n == 0:
        return float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + margin) / denom


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Cohen's kappa for paired binary ratings."""
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    p_a_yes = sum(1 for a, _ in pairs if a) / n
    p_b_yes = sum(1 for _, b in pairs if b) / n
    pe = p_a_yes * p_b_yes + (1 - p_a_yes) * (1 - p_b_yes)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def _load_records() -> list[dict]:
    """Read all records of the audit log."""
    if not LOG_PATH.exists():
        return []
    out = []
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _dedupe_last(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Map (auditor_id, session_id) to the latest record (a resubmission replaces the earlier one)."""
    out: dict[tuple[str, str], dict] = {}
    for r in records:
        out[(r["auditor_id"], r["session_id"])] = r
    return out


def main() -> None:
    """Print progress and, unless --progress-only, compute the summary statistics and write the CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else None
    total_cases = manifest["total"] if manifest else None

    records = _load_records()
    dedup = _dedupe_last(records)

    # Group the latest records by auditor
    per_auditor: dict[str, list[dict]] = defaultdict(list)
    for (auditor, _session), rec in dedup.items():
        per_auditor[auditor].append(rec)

    print(f"Total log rows (including revisions): {len(records)}")
    print(f"Unique audited cases (auditor,session): {len(dedup)}")
    for auditor, recs in sorted(per_auditor.items()):
        done = len(recs)
        flagged = sum(1 for r in recs if r["has_hallucination"])
        pct = f"{done}/{total_cases}" if total_cases else str(done)
        print(f"  {auditor}: {pct} cases, {flagged} flagged as hallucination")

    if args.progress_only:
        return

    # Hallucination rate with a Wilson upper bound
    all_recs = [r for recs in per_auditor.values() for r in recs]
    if not all_recs:
        print("No data to summarize yet.")
        return

    n = len(all_recs)
    n_flagged = sum(1 for r in all_recs if r["has_hallucination"])
    hr_human = n_flagged / n
    hr_upper = wilson_upper(n_flagged, n)

    agree = disagree_human_only = disagree_guardrail_only = 0
    cat_counter: Counter[str] = Counter()
    durations = [r["duration_ms"] for r in all_recs if r.get("duration_ms")]

    for r in all_recs:
        cat_counter.update(r.get("categories", []))

    # Median review time per case
    median_s = None
    if durations:
        s = sorted(durations)
        mid = len(s) // 2
        median_ms = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
        median_s = median_ms / 1000

    # Summary table rows
    rows = [
        {"metric": "n_cases_audited", "value": n},
        {"metric": "n_auditors", "value": len(per_auditor)},
        {"metric": "HR_human_pct", "value": round(hr_human * 100, 2)},
        {"metric": "HR_human_wilson_upper_pct", "value": round(hr_upper * 100, 2)},
        {"metric": "median_duration_s", "value": round(median_s, 1) if median_s else ""},
    ]
    for cat, cnt in cat_counter.most_common():
        rows.append({"metric": f"category__{cat}", "value": cnt})

    # Inter-auditor agreement when at least two auditors share sessions
    auditors = sorted(per_auditor.keys())
    if len(auditors) >= 2:
        a_recs = {r["session_id"]: r["has_hallucination"] for r in per_auditor[auditors[0]]}
        b_recs = {r["session_id"]: r["has_hallucination"] for r in per_auditor[auditors[1]]}
        overlap = sorted(set(a_recs) & set(b_recs))
        pairs = [(a_recs[s], b_recs[s]) for s in overlap]
        if pairs:
            kappa = cohens_kappa(pairs)
            rows.append({"metric": f"kappa_{auditors[0]}_vs_{auditors[1]}_n{len(pairs)}", "value": round(kappa, 3)})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nHuman HR: {hr_human*100:.2f}% (Wilson upper {hr_upper*100:.2f}%), n={n}")
    if median_s:
        print(f"Median time/case: {median_s:.1f}s")
    print(f"Written: {OUT_CSV}")


if __name__ == "__main__":
    main()
