#!/usr/bin/env python3
"""Edge-latency measurement of the Variant C narrative pipeline (local Ollama on the target laptop).

Purpose: re-measure the per-narrative wall-clock latency of the four compared LLMs on the edge
configuration (MacBook Air M3, 16 GB), with the current prompts and guardrail, and with a warm-up so
that model loading and first-call effects do not enter the timed generations.

Design
  * Same call as `evaluate_llm_paper.eval_one`: `generate_from_facts(fc, model, "c")`, timed with
    `time.perf_counter`. The time therefore includes the regeneration and the template fallback when
    they occur, exactly like the cloud-pod `wall_s` of T1/T2.
  * Warm-up: before the timed generations of each model, `--warmup` untimed generations run on
    sessions that are NOT in the timed sample. They load the model into memory and warm the Metal
    kernels and the prompt cache. Timed rows with `load_duration_s` above `--cold-threshold` are
    flagged `cold=true` and counted in the summary.
  * The timed sample is the same set of sessions for every model (paired), stratified by
    `argmax_class` x `ood_status` and drawn with a fixed seed.
  * Models run one at a time. After each model the script stops it (`ollama stop`) and waits
    `--cooldown` seconds, because the fanless MacBook Air throttles under sustained load.
  * Output rows carry a hashed session label, never the session ID.
  * Never pulls a model and never calls a cloud API; it refuses a non-local `OLLAMA_HOST`.

Usage (run it yourself; a full run takes about an hour for n=30)
    export OLLAMA_KEEP_ALIVE=30m          # keep the loaded model resident between calls
    # plug in the charger, close other apps, disable low-power mode
    python -m llm_xai.tools.measure_edge_latency --dry-run                  # plan only, no LLM call
    python -m llm_xai.tools.measure_edge_latency --n 30 --warmup 2          # the measurement
    python -m llm_xai.tools.measure_edge_latency --summarize-only           # re-aggregate raw rows

Outputs (llm_xai/results/edge_latency/): edge_latency_raw.jsonl, edge_latency_summary.csv,
edge_latency_env.json. `--resume` (default) skips model/session pairs already in the raw file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llm_xai.backend import RESULTS_DIR  # noqa: E402
from llm_xai.backend.llm_engine import OLLAMA_HOST, FactContext, run_manifest  # noqa: E402

GOLD_DIR = RESULTS_DIR / "gold_facts"
OUT_DIR = RESULTS_DIR / "edge_latency"
RAW_FP = OUT_DIR / "edge_latency_raw.jsonl"
SUMMARY_FP = OUT_DIR / "edge_latency_summary.csv"
ENV_FP = OUT_DIR / "edge_latency_env.json"

DEFAULT_MODELS = ["llama31_8b", "mistral7b", "biomistral", "qwen25_14b"]


def sid_label(session_id: str) -> str:
    """Short irreversible label of a session ID, so raw rows never carry the ID."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:10]


def load_gold() -> list[FactContext]:
    """Load every gold fact context (the 232-session test fold)."""
    return [FactContext(**json.loads(f.read_text())) for f in sorted(GOLD_DIR.glob("*.json"))]


def stratum(fc: FactContext) -> tuple[str, str]:
    """Stratum of a session: (argmax class, OOD status)."""
    return (fc.prediction_anchor.argmax_class, fc.prediction_anchor.ood_status)


def draw_sessions(gold: list[FactContext], n_timed: int, n_warm: int, seed: int):
    """Return (warmup_sessions, timed_sessions), disjoint, round-robin over strata with a fixed seed."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[FactContext]] = defaultdict(list)
    for fc in gold:
        groups[stratum(fc)].append(fc)
    for g in groups.values():
        rng.shuffle(g)
    order: list[FactContext] = []
    keys = sorted(groups)
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                order.append(groups[k].pop())
    if n_timed + n_warm > len(order):
        raise SystemExit(f"requested {n_timed}+{n_warm} sessions but only {len(order)} are available")
    # Timed sample first so that it does not depend on the warm-up size
    return order[n_timed:n_timed + n_warm], order[:n_timed]


def _sh(cmd: list[str]) -> str:
    """Run a command and return its stripped stdout ('' when unavailable)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment() -> dict:
    """Hardware and software context recorded next to the measurements."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ram_gb": round(int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1),
        "power": _sh(["pmset", "-g", "batt"]).splitlines()[:1],
        "low_power_mode": _sh(["pmset", "-g"]).count("lowpowermode         1") > 0,
        "ollama_version": _sh(["ollama", "--version"]),
        "ollama_host": OLLAMA_HOST,
        "ollama_keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "(unset: Ollama default 5m)"),
        "python": platform.python_version(),
        **run_manifest(),
    }


def check_ready(models_cfg: dict, models: list[str], allow_non_mac: bool) -> None:
    """Refuse a non-local host or a non-Mac machine, and check the models are already pulled."""
    host = urlparse(OLLAMA_HOST).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1"):
        raise SystemExit(f"OLLAMA_HOST={OLLAMA_HOST} is not local; edge latency must be measured locally.")
    if platform.system() != "Darwin" and not allow_non_mac:
        raise SystemExit("This is not macOS; the edge target is the MacBook Air M3 (use --allow-non-mac to override).")
    listed = _sh(["ollama", "list"])
    if not listed:
        raise SystemExit("`ollama list` returned nothing; start `ollama serve` first.")
    missing = []
    for m in models:
        entry = models_cfg.get(m)
        if entry is None or entry.get("kind") != "ollama":
            raise SystemExit(f"model key {m!r} is not a local Ollama model in configs/llm_models.yaml")
        if entry["model"].split(":")[0] not in listed:
            missing.append(entry["model"])
    if missing:
        raise SystemExit("Models not pulled (this script never downloads): " + ", ".join(missing))


def read_raw() -> list[dict]:
    """Read the raw rows written so far."""
    if not RAW_FP.exists():
        return []
    return [json.loads(line) for line in RAW_FP.read_text().splitlines() if line.strip()]


def run_one(fc: FactContext, model_key: str, cold_threshold: float) -> dict:
    """One timed generation, the same call as `evaluate_llm_paper.eval_one`."""
    from llm_xai.backend.pipeline import generate_from_facts  # imported lazily (not needed by --dry-run)

    t0 = time.perf_counter()
    res = generate_from_facts(fc, model_key, "c")
    wall = time.perf_counter() - t0
    g = res.get("guardrail")
    meta = res.get("llm_meta", {}) or {}
    tok = meta.get("eval_count") or 0
    dur_ns = meta.get("eval_duration_ns") or 0
    load_s = round((meta.get("load_duration_ns") or 0) / 1e9, 3)
    return {
        "model": model_key,
        "sid": sid_label(fc.session_id),
        "argmax_class": fc.prediction_anchor.argmax_class,
        "ood_status": fc.prediction_anchor.ood_status,
        "wall_s": round(wall, 2),
        "llm_calls": res.get("llm_calls", 1),
        "source": g["source"] if g else "parse_error",
        "passed": bool(g["passed"]) if g else False,
        "tokens": tok,
        "ms_per_token": round(dur_ns / 1e6 / tok, 2) if tok else None,
        "load_duration_s": load_s,
        "cold": load_s > cold_threshold,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def stop_model(models_cfg: dict, model_key: str) -> None:
    """Unload a model from memory (best effort)."""
    _sh(["ollama", "stop", models_cfg[model_key]["model"]])


def bootstrap_median_ci(xs: list[float], reps: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% interval of the median."""
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(reps))
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def quartiles(xs: list[float]) -> tuple[float, float]:
    """First and third quartile (inclusive method)."""
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0], q[2]


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate raw rows per model into the summary table."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = []
    for m in DEFAULT_MODELS + sorted(set(by_model) - set(DEFAULT_MODELS)):
        rs = by_model.get(m)
        if not rs:
            continue
        w = [r["wall_s"] for r in rs]
        first = [r["wall_s"] for r in rs if r["llm_calls"] == 1]
        mpt = [r["ms_per_token"] for r in rs if r["ms_per_token"]]
        q1, q3 = quartiles(w) if len(w) >= 2 else (w[0], w[0])
        lo, hi = bootstrap_median_ci(w) if len(w) >= 2 else (w[0], w[0])
        out.append({
            "model": m, "n": len(rs),
            "wall_s_median": round(statistics.median(w), 2),
            "wall_s_q1": round(q1, 2), "wall_s_q3": round(q3, 2),
            "wall_s_median_ci95_lo": round(lo, 2), "wall_s_median_ci95_hi": round(hi, 2),
            "wall_s_mean": round(statistics.mean(w), 2),
            "wall_s_median_single_call_only": round(statistics.median(first), 2) if first else "",
            "ms_per_token_median": round(statistics.median(mpt), 2) if mpt else "",
            "regenerated_pct": round(100 * sum(r["llm_calls"] == 2 for r in rs) / len(rs), 2),
            "fallback_pct": round(100 * sum(r["source"] == "deterministic_fallback" for r in rs) / len(rs), 2),
            "cold_start_rows": sum(bool(r["cold"]) for r in rs),
        })
    return out


def write_summary(rows: list[dict]) -> None:
    """Write the summary CSV and print it."""
    table = summarize(rows)
    if not table:
        print("[edge] no rows to summarize")
        return
    with SUMMARY_FP.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"[edge] wrote {SUMMARY_FP}")
    for r in table:
        print(f"  {r['model']:12s} n={r['n']:3d} median={r['wall_s_median']:7.2f}s "
              f"IQR=[{r['wall_s_q1']}, {r['wall_s_q3']}] CI95=[{r['wall_s_median_ci95_lo']}, "
              f"{r['wall_s_median_ci95_hi']}] regen={r['regenerated_pct']}% fallback={r['fallback_pct']}% "
              f"cold={r['cold_start_rows']}")


def main() -> int:
    """Parse arguments, run the warm-up and the timed generations per model, then summarize."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated model keys, in run order")
    ap.add_argument("--n", type=int, default=30, help="timed sessions per model (same sessions for every model)")
    ap.add_argument("--warmup", type=int, default=2, help="untimed warm-up generations per model")
    ap.add_argument("--cooldown", type=int, default=120, help="seconds to wait between models (thermal)")
    ap.add_argument("--cold-threshold", type=float, default=2.0, help="load_duration_s above this flags a cold row")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true", help="ignore existing raw rows and start a new file")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no LLM call, nothing written")
    ap.add_argument("--summarize-only", action="store_true", help="only re-aggregate the existing raw rows")
    ap.add_argument("--allow-non-mac", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        write_summary(read_raw())
        return 0

    from llm_xai.backend.pipeline import load_models_cfg  # light; reads configs/llm_models.yaml

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    models_cfg = load_models_cfg()
    gold = load_gold()
    warm, timed = draw_sessions(gold, args.n, args.warmup, args.seed)
    strata = defaultdict(int)
    for fc in timed:
        strata[stratum(fc)] += 1
    print(f"[edge] {len(gold)} gold sessions; timed sample n={len(timed)} per model, warm-up={len(warm)} per model")
    print(f"[edge] strata of the timed sample: {dict(sorted(strata.items()))}")
    print(f"[edge] models in run order: {models}; cooldown {args.cooldown}s")
    if args.dry_run:
        print("[edge] dry run: no model was called")
        return 0

    check_ready(models_cfg, models, args.allow_non_mac)
    if args.no_resume and RAW_FP.exists():
        RAW_FP.rename(RAW_FP.with_suffix(f".{datetime.now():%Y%m%d%H%M%S}.bak"))
    done = {(r["model"], r["sid"]) for r in read_raw()}
    ENV_FP.write_text(json.dumps(environment(), indent=2))
    print(f"[edge] environment written to {ENV_FP}")

    for mi, model_key in enumerate(models):
        todo = [fc for fc in timed if (model_key, sid_label(fc.session_id)) not in done]
        if not todo:
            print(f"[edge] {model_key}: already complete, skipping")
            continue
        print(f"[edge] {model_key}: warm-up ({len(warm)} untimed generations)")
        for fc in warm:
            run_one(fc, model_key, args.cold_threshold)  # result discarded
        for i, fc in enumerate(todo, 1):
            row = run_one(fc, model_key, args.cold_threshold)
            row["order"] = i
            with RAW_FP.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {model_key} {i:3d}/{len(todo)} wall={row['wall_s']:6.2f}s calls={row['llm_calls']} "
                  f"src={row['source']}{' COLD' if row['cold'] else ''}")
        stop_model(models_cfg, model_key)
        if mi < len(models) - 1 and args.cooldown > 0:
            print(f"[edge] cooldown {args.cooldown}s")
            time.sleep(args.cooldown)

    write_summary(read_raw())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
