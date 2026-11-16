"""Build the case order of the blind human audit of llama31_8b variant C narratives.

Inputs (read directly):
  - results/eval_raw/<session_id>__llama31_8b__c.json  narrative and guardrail verdict
  - results/gold_facts/<session_id>.json               full fact context

Cases are interleaved round-robin over strata (argmax_class x ood_status) with a fixed seed, so
any prefix of the order stays representative of all strata.

Run (writes cases_manifest.json; re-run when the eval_raw session list changes):
    python3 audit/build_manifest.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_RAW = ROOT / "results" / "eval_raw"
GOLD_FACTS = ROOT / "results" / "gold_facts"
OUT_PATH = Path(__file__).resolve().parent / "cases_manifest.json"

MODEL = "llama31_8b"
VARIANT = "c"
SEED = 42


def _round_robin_by_stratum(strata: dict[tuple, list[str]], rng: random.Random) -> list[str]:
    """Shuffle within each stratum, then take one session per stratum in turn."""
    for ids in strata.values():
        rng.shuffle(ids)
    keys = sorted(strata.keys())
    rng.shuffle(keys)
    queues = {k: list(v) for k, v in strata.items()}
    order: list[str] = []
    while any(queues[k] for k in keys):
        for k in keys:
            if queues[k]:
                order.append(queues[k].pop(0))
    return order


def main() -> None:
    """Collect the audited sessions, order them, and write the manifest."""
    suffix = f"__{MODEL}__{VARIANT}.json"
    files = sorted(EVAL_RAW.glob(f"*{suffix}"))
    if not files:
        raise SystemExit(f"No {suffix} files found in {EVAL_RAW}")

    # Group sessions by (predicted class, out-of-domain status)
    strata: dict[tuple, list[str]] = defaultdict(list)
    meta: dict[str, dict] = {}
    missing_gold = []
    for f in files:
        session_id = f.name[: -len(suffix)]
        gold_path = GOLD_FACTS / f"{session_id}.json"
        if not gold_path.exists():
            missing_gold.append(session_id)
            continue
        gold = json.loads(gold_path.read_text())
        pa = gold["prediction_anchor"]
        stratum = (pa["argmax_class"], pa["ood_status"])
        strata[stratum].append(session_id)
        meta[session_id] = {
            "argmax_class": pa["argmax_class"],
            "ood_status": pa["ood_status"],
            "fact_digest": gold.get("fact_digest"),
        }

    if missing_gold:
        raise SystemExit(f"gold_facts missing for {len(missing_gold)} sessions: {missing_gold[:5]}...")

    rng = random.Random(SEED)
    order = _round_robin_by_stratum(strata, rng)

    cases = [
        {
            "order_index": i,
            "session_id": sid,
            "argmax_class": meta[sid]["argmax_class"],
            "ood_status": meta[sid]["ood_status"],
            "fact_digest": meta[sid]["fact_digest"],
        }
        for i, sid in enumerate(order)
    ]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "variant": VARIANT,
        "seed": SEED,
        "ordering": "round_robin_by_stratum(argmax_class, ood_status)",
        "total": len(cases),
        "stratum_sizes": {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(strata.items())},
        "cases": cases,
    }
    OUT_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {OUT_PATH} ({len(cases)} cases)")
    for k, v in manifest["stratum_sizes"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
