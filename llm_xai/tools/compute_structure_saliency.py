#!/usr/bin/env python3
"""Anatomical grounding accuracy (AGA): per-structure share of saliency energy.

For each test session this recomputes the HiResCAM maps (target = predicted class, as in
`build_gold_facts.py`) and splits their energy over individual structures of the FreeSurfer
`aseg.mgz`: hippocampus, ventricles and amygdala. Energy is the brain-masked attribution energy
inside the structure (Energy-based Pointing Game, a generalisation of `_on_target_pct` in
`backend/inference_engine.py`), averaged over slices weighted by MIL attention. The result is
compared with the uniform baseline (voxel fraction) as a concentration ratio.

Why not `dominant_structures` of the fact context: it ranks structures by voxel count and is
almost constant across patients, so it is not a per-session saliency signal.

The script is a decoupled reporting metric. It does not modify the inference engine, the gold
facts, the prompts or the fact digest; outputs go to `results/aga_raw/` and
`results/AGA_structure_summary.csv`, joined to sessions by `session_id`.

Limitations:
  - Only `aseg.mgz` (subcortical) is available, so only structures it resolves can be evaluated;
    cortical regions (parahippocampal gyrus, entorhinal cortex) and thalamus cannot.
  - `_EXPECTED_BY_CLASS` (hippocampus and ventricles for AD and MCI, no expectation for CN) is a
    methodological choice, not a validated ground truth.

    python -m llm_xai.tools.compute_structure_saliency --limit 15   # pilot
    python -m llm_xai.tools.compute_structure_saliency               # all test sessions
    python -m llm_xai.tools.compute_structure_saliency --resume      # continue an interrupted run
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Engine functions are used as they are; the inference engine is not modified
from backend.inference_engine import (  # noqa: E402
    CLASS_NAMES, MODEL_READY, ROI_LABELS,
    _PLANE_KEYS, _load_aseg_for_session, _load_session_tensors,
    _read_tabular_from_meta, _slice_from_aseg, cam_attr, device, model,
)
from llm_xai.backend import RESULTS_DIR  # noqa: E402
from llm_xai.backend.volebpg_aggregate import _local_index  # noqa: E402
from llm_xai.tools.build_gold_facts import resolve_test_sessions  # noqa: E402
from llm_xai.tools.stats_utils import wilson_ci  # noqa: E402

AGA_METHOD_VERSION = "0.1.0"

# Individual structures resolvable from aseg.mgz. Cortex (ids 3, 42) is excluded: it covers most
# of the brain, so it would dominate the energy split without anatomical meaning.
AGA_STRUCT_IDS: dict[str, list[int]] = {
    "hippocampus": ROI_LABELS["hippocampus"],
    "ventricles": ROI_LABELS["ventricles"],
    "amygdala": ROI_LABELS["amygdala"],
}

# Structures expected to dominate saliency per predicted class (no expectation for CN)
_EXPECTED_BY_CLASS: dict[str, list[str] | None] = {
    "AD": ["hippocampus", "ventricles"],
    "MCI": ["hippocampus", "ventricles"],
    "CN": None,
}


def _structure_energy_one_slice(
    aseg_slice: np.ndarray, cam_slice: np.ndarray, brain_mask: np.ndarray,
) -> dict[str, float]:
    """Percentage of brain-masked attribution energy inside each structure, for one slice.

    Same construction as `_on_target_pct` in inference_engine.py, applied per structure; the
    remainder falls on other tissue.
    """
    masked = cam_slice * brain_mask
    denom = float(np.sum(masked)) + 1e-8
    out = {}
    for name, ids in AGA_STRUCT_IDS.items():
        target = np.isin(aseg_slice, ids).astype(np.float32)
        out[name] = float(np.sum(masked * target) / denom) * 100.0
    return out


def _structure_baseline_one_slice(aseg_slice: np.ndarray) -> tuple[dict[str, float], int]:
    """Voxel percentage of the brain occupied by each structure (the uniform baseline), and the brain voxel count."""
    brain = int((aseg_slice > 0).sum())
    if brain == 0:
        return {name: 0.0 for name in AGA_STRUCT_IDS}, 0
    out = {}
    for name, ids in AGA_STRUCT_IDS.items():
        out[name] = float(np.isin(aseg_slice, ids).sum()) / brain * 100.0
    return out, brain


def compute_session_structure_saliency(session_id: str) -> dict:
    """Compute per-structure saliency energy, baseline and concentration ratio for one session.

    Reproduces the primary decision path of `predict_session` (image-only argmax, HiResCAM) so
    the predicted class and CAM target match the gold facts. Slice values are averaged with MIL
    attention weights. Also reports whether the top structure by energy is an expected one for the
    predicted class (`aga_hit`).
    """
    t0 = time.perf_counter()
    vol_tensor_stack, tensors, pil_images, session_dir, canonical_id = _load_session_tensors(session_id)
    vol_tensor_batch = vol_tensor_stack.unsqueeze(0).to(device)

    tab_list, _tab_extras = _read_tabular_from_meta(session_id, canonical_id, True, True, None, None)
    tab_tensor_real = torch.tensor([tab_list], dtype=torch.float32).to(device)
    # Image-only decision path: covariates zeroed, as in predict_session
    tab_tensor = torch.zeros_like(tab_tensor_real)

    with torch.no_grad():
        volume_rep, attn_weights_raw = model.encode_volume(vol_tensor_batch)
        t_emb = model.tabular_proj(tab_tensor)
        logits = model.classifier(torch.cat([volume_rep, t_emb], dim=-1))
        probs = torch.softmax(logits, dim=1)[0]
        attns = attn_weights_raw[0].cpu().numpy()
    pred_id = int(torch.argmax(probs).item())
    argmax_class = CLASS_NAMES[pred_id]

    cams = cam_attr(model, vol_tensor_stack, tab_tensor[0].cpu(), pred_id,
                     layer="encoder.layer.3", mode="hirescam")

    aseg_vol = _load_aseg_for_session(session_id, canonical_id)
    if aseg_vol is None:
        raise RuntimeError(f"{session_id}: aseg.mgz not found (required for the test sessions)")

    n_slices = cams.shape[0]
    per_slice_energy: list[dict[str, float]] = []
    per_slice_baseline: list[dict[str, float]] = []
    w_by_idx: dict[int, float] = {}
    for i in range(n_slices):
        img_np = np.array(pil_images[i]).astype(np.float32) / 255.0
        brain_mask = (img_np.mean(axis=2) > 0.04).astype(np.float32)
        plane_key = _PLANE_KEYS[i]
        local_idx = _local_index(i)
        aseg_slice = _slice_from_aseg(aseg_vol, plane_key, local_idx)
        per_slice_energy.append(_structure_energy_one_slice(aseg_slice, cams[i], brain_mask))
        base, _brain = _structure_baseline_one_slice(aseg_slice)
        per_slice_baseline.append(base)
        w_by_idx[i] = float(attns[i]) * 100.0

    wsum = sum(w_by_idx.values()) or float(n_slices)
    structures: dict[str, dict[str, float | None]] = {}
    for name in AGA_STRUCT_IDS:
        energy = sum(per_slice_energy[i][name] * w_by_idx.get(i, 1.0) for i in range(n_slices)) / wsum
        baseline = float(np.mean([per_slice_baseline[i][name] for i in range(n_slices)]))
        ratio = round(energy / baseline, 3) if baseline > 1e-9 else None
        structures[name] = {
            "energy_pct": round(energy, 3),
            "baseline_uniform_pct": round(baseline, 3),
            "concentration_ratio": ratio,
        }

    top_structure = max(structures, key=lambda k: structures[k]["energy_pct"])
    expected = _EXPECTED_BY_CLASS.get(argmax_class)
    aga_hit = (top_structure in expected) if expected else None

    return {
        "session_id": session_id,
        "canonical_session_id": canonical_id,
        "argmax_class": argmax_class,
        "xai_method": "hirescam",
        "aga_method_version": AGA_METHOD_VERSION,
        "structures": structures,
        "top_structure_by_energy": top_structure,
        "expected_structures": expected,
        "aga_hit": aga_hit,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def write_summary(raw_dir: Path, out_csv: Path) -> None:
    """Aggregate the per-session results by predicted class into the summary CSV, with Wilson intervals of the hit rate."""
    from collections import defaultdict
    import statistics

    rows = [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*.json"))]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["argmax_class"]].append(r)

    out_rows = []
    for cls in sorted(by_class):
        rs = by_class[cls]
        n = len(rs)
        row = {"argmax_class": cls, "n": n}
        for name in AGA_STRUCT_IDS:
            energies = [r["structures"][name]["energy_pct"] for r in rs]
            ratios = [r["structures"][name]["concentration_ratio"] for r in rs
                      if r["structures"][name]["concentration_ratio"] is not None]
            row[f"{name}_energy_pct_mean"] = round(statistics.mean(energies), 3) if energies else None
            row[f"{name}_energy_pct_median"] = round(statistics.median(energies), 3) if energies else None
            row[f"{name}_concentration_ratio_mean"] = round(statistics.mean(ratios), 3) if ratios else None
        hits = [r["aga_hit"] for r in rs if r["aga_hit"] is not None]
        if hits:
            k = sum(1 for h in hits if h)
            lo, hi = wilson_ci(k, len(hits))
            row["aga_hit_rate_pct"] = round(k / len(hits) * 100, 2)
            row["aga_hit_wilson_lo_pct"] = round(lo * 100, 2)
            row["aga_hit_wilson_hi_pct"] = round(hi * 100, 2)
            row["aga_hit_k"] = k
            row["aga_hit_n"] = len(hits)
        else:
            row["aga_hit_rate_pct"] = row["aga_hit_wilson_lo_pct"] = row["aga_hit_wilson_hi_pct"] = None
            row["aga_hit_k"] = row["aga_hit_n"] = 0
        out_rows.append(row)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"[compute_structure_saliency] summary -> {out_csv}")
    for row in out_rows:
        print(f"  {row['argmax_class']}: n={row['n']}  "
              f"hippocampus_energy%={row['hippocampus_energy_pct_mean']}  "
              f"hit_rate%={row['aga_hit_rate_pct']} ({row['aga_hit_k']}/{row['aga_hit_n']})")


def main() -> int:
    """Compute per-session structure saliency (unless --summarize-only) and write the summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 232 sessions")
    ap.add_argument("--sessions-file", default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "aga_raw"))
    ap.add_argument("--summary-out", default=str(RESULTS_DIR / "AGA_structure_summary.csv"),
                    help="summary CSV (override for pilots so the real results are not overwritten)")
    ap.add_argument("--resume", action="store_true", help="skip sessions whose file already exists")
    ap.add_argument("--summarize-only", action="store_true", help="skip computation, just rewrite the summary CSV")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        if not MODEL_READY:
            print("[compute_structure_saliency] MODEL_READY=False -- the vision model failed to load.", file=sys.stderr)
            return 1
        sessions = resolve_test_sessions(args.sessions_file)
        if args.limit:
            sessions = sessions[: args.limit]
        print(f"[compute_structure_saliency] {len(sessions)} sessions (AGA_METHOD_VERSION={AGA_METHOD_VERSION}).")

        n_ok = n_skip = 0
        for i, sid in enumerate(sessions, 1):
            fp = out_dir / f"{sid}.json"
            if args.resume and fp.exists():
                n_skip += 1
                continue
            try:
                result = compute_session_structure_saliency(sid)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(sessions)}] {sid}  SKIP ({e})")
                continue
            fp.write_text(json.dumps(result, indent=2))
            n_ok += 1
            print(f"  [{i}/{len(sessions)}] {sid}  ok  top={result['top_structure_by_energy']}  "
                  f"hit={result['aga_hit']}  ({result['wall_s']}s)")
        print(f"[compute_structure_saliency] done: {n_ok} new, {n_skip} skipped (resume) -> {out_dir}")

    write_summary(out_dir, Path(args.summary_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
