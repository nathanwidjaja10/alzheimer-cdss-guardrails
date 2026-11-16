#!/usr/bin/env python3
"""Build the deterministic "gold" fact context of every internal test session.

For each test session: predict_session (HiResCAM) -> build_fact_context -> write
`llm_xai/results/gold_facts/<session>.json` and `gold_facts_index.csv`. All paper metrics are
computed against these files.

Test sessions: sessions in the clinical catalogue that have brain.mgz and aseg.mgz, kept when
their label is the subject's highest severity, and whose subject is in `split_manifest['test']`
(232 sessions: 183 CN, 37 MCI, 12 AD). The count is printed; use `--sessions-file` to override
the list.

    python -m llm_xai.tools.build_gold_facts --limit 5      # quick check
    python -m llm_xai.tools.build_gold_facts                # full run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.inference_engine import predict_session  # noqa: E402
from llm_xai.backend import RESULTS_DIR  # noqa: E402
from llm_xai.backend.llm_engine import build_fact_context  # noqa: E402

CLINICAL_JSON = REPO_ROOT / "oasis_data" / "clinical_metadata.json"
FREESURFER_DIR = REPO_ROOT / "oasis_data" / "freesurfer_data"
SPLIT_MANIFEST = REPO_ROOT / "checkpoints" / "split_manifest.json"

LABEL_PRIORITY = {"CN": 0, "MCI": 1, "AD": 2}


def map_cdr_to_label(dx1: str, cdr: str) -> str | None:
    """Map CDR (or, when it is missing, the free-text diagnosis) to CN / MCI / AD, as in the training notebook."""
    try:
        cdr_val = float(cdr)
    except (TypeError, ValueError):
        d = str(dx1).lower().strip()
        if "normal" in d:
            return "CN"
        if any(k in d for k in ("mci", "mild cognitive", "impairment")):
            return "MCI"
        if any(k in d for k in ("alzheimer", "dementia", "ad")):
            return "AD"
        return None
    if cdr_val == 0:
        return "CN"
    if cdr_val == 0.5:
        return "MCI"
    if cdr_val >= 1.0:
        return "AD"
    return None


def build_catalog() -> list[dict]:
    """List labelled sessions that have both brain.mgz and aseg.mgz."""
    meta = json.loads(CLINICAL_JSON.read_text())
    rows: list[dict] = []
    for sid, info in meta.items():
        brain = FREESURFER_DIR / sid / "brain.mgz"
        if not brain.exists():
            brain = FREESURFER_DIR / sid / "mri" / "brain.mgz"
        if not brain.exists():
            continue
        if not (brain.parent / "aseg.mgz").exists():
            continue
        label = map_cdr_to_label(info.get("diagnosis", ""), info.get("cdr", ""))
        if label is None:
            continue
        rows.append({
            "session_id": sid,
            "subject_id": info.get("subject_id", sid.split("_")[0]),
            "label": label,
        })
    return rows


def filter_consistent_sessions(rows: list[dict]) -> list[dict]:
    """Keep only sessions labelled with the highest severity for their subject."""
    max_lp: dict[str, int] = {}
    for r in rows:
        max_lp[r["subject_id"]] = max(max_lp.get(r["subject_id"], -1), LABEL_PRIORITY[r["label"]])
    return [r for r in rows if LABEL_PRIORITY[r["label"]] == max_lp[r["subject_id"]]]


def resolve_test_sessions(sessions_file: str | None) -> list[str]:
    """Return the test session ids from an explicit file (.txt or .json) or from the split manifest."""
    if sessions_file:
        p = Path(sessions_file)
        if p.suffix == ".json":
            return list(json.loads(p.read_text()))
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    manifest = json.loads(SPLIT_MANIFEST.read_text())
    test_subjects = set(manifest["test"])
    rows = filter_consistent_sessions(build_catalog())
    sess = sorted(r["session_id"] for r in rows if r["subject_id"] in test_subjects)
    return sess


def main() -> int:
    """Generate (or reuse) the fact context of each session and write the index."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sessions-file", default=None, help="override the session list (.txt/.json)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "gold_facts"))
    ap.add_argument("--resume", action="store_true", help="skip sessions whose file already exists")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = resolve_test_sessions(args.sessions_file)
    if args.limit:
        sessions = sessions[: args.limit]
    print(f"[build_gold_facts] {len(sessions)} sessions (expected full test set: 232).")

    index_rows: list[dict] = []
    for i, sid in enumerate(sessions, 1):
        fp = out_dir / f"{sid}.json"
        if args.resume and fp.exists():
            fc = json.loads(fp.read_text())
        else:
            t0 = time.perf_counter()
            try:
                ai = predict_session(sid, xai_method="hirescam")
                facts = build_fact_context(ai, sid)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(sessions)}] {sid}  SKIP ({e})")
                continue
            fc = facts.model_dump()
            fp.write_text(json.dumps(fc, indent=2, default=str))
            print(f"  [{i}/{len(sessions)}] {sid}  ok  ({time.perf_counter() - t0:.1f}s)")
        pa, sx = fc["prediction_anchor"], fc["spatial_xai_anchor"]
        index_rows.append({
            "session_id": sid,
            "argmax_class": pa["argmax_class"],
            "ood_status": pa["ood_status"],
            "aseg_available": sx["aseg_available"],
            "primary_plane": sx["primary_plane"],
            "fact_digest": fc["fact_digest"],
        })

    idx_path = RESULTS_DIR / "gold_facts_index.csv"
    with open(idx_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session_id", "argmax_class", "ood_status",
                                          "aseg_available", "primary_plane", "fact_digest"])
        w.writeheader()
        w.writerows(index_rows)

    from collections import Counter

    print(f"[build_gold_facts] done: {len(index_rows)} facts -> {out_dir}")
    print("  class :", dict(Counter(r["argmax_class"] for r in index_rows)))
    print("  ood   :", dict(Counter(r["ood_status"] for r in index_rows)))
    print("  aseg  :", dict(Counter(r["aseg_available"] for r in index_rows)))
    print(f"  index : {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
