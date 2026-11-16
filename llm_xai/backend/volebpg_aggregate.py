"""Volume-level aggregation of slice-level XAI outputs for the fact context.

`predict_session()` returns per-slice attention weights and per-slice on-target percentages
(None without a segmentation). This module aggregates them into the volume-level quantities of
the `spatial_xai_anchor` block:
  - `primary_plane` and `plane_attention_share_pct`
  - `on_target_energy_pct` (attention-weighted mean over slices)
  - `baseline_uniform_pct` (target-voxel fraction of the brain, from aseg.mgz)
  - `ebpg_concentration_ratio`
  - `dominant_structures` (voxel-count proxy, used as fallback)

The segmentation is loaded through the helpers of `backend.inference_engine`, which is not modified.
"""
from __future__ import annotations

import numpy as np

try:
    from backend.inference_engine import (
        ROI_LABELS, N_AXIAL, N_CORONAL, _PLANE_KEYS,
        _load_aseg_for_session, _slice_from_aseg,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.inference_engine import (
        ROI_LABELS, N_AXIAL, N_CORONAL, _PLANE_KEYS,
        _load_aseg_for_session, _slice_from_aseg,
    )

# Structure names and their FreeSurfer label ids (a subset of inference_engine.ROI_LABELS)
_STRUCT_IDS = {
    "hippocampus": [17, 53],
    "lateral ventricle": [4, 43],
    "inferior lateral ventricle": [5, 44],
    "third ventricle": [14],
    "amygdala": [18, 54],
}


def _local_index(i: int) -> int:
    """Index of a slice within its plane (mirrors the rule in inference_engine.predict_session)."""
    if i < N_AXIAL:
        return i
    if i < N_AXIAL + N_CORONAL:
        return i - N_AXIAL
    return i - N_AXIAL - N_CORONAL


def aggregate_plane_attention(attention_weights: list[dict]) -> dict:
    """Aggregate slice attention into per-plane shares and the three top slices.

    `attention_weights` is `ai_result['attention_weights']` (elements: index, plane, weight,
    percentage).
    """
    groups: dict[str, float] = {}
    for a in attention_weights:
        plane = str(a.get("plane", "")).split()[0].capitalize() or "Unknown"
        groups[plane] = groups.get(plane, 0.0) + float(a.get("percentage", float(a.get("weight", 0.0)) * 100.0))
    total = sum(groups.values()) or 1.0
    shares = {k: round(v / total * 100.0, 1) for k, v in groups.items()}
    primary = max(shares, key=shares.get) if shares else None
    ranked = sorted(attention_weights, key=lambda a: -float(a.get("percentage", 0.0)))
    top = [
        {
            "name": a.get("plane"),
            "index": int(a.get("index", -1)),
            "attention_share_pct": round(float(a.get("percentage", 0.0)), 1),
        }
        for a in ranked[:3]
    ]
    return {"primary_plane": primary, "plane_attention_share_pct": shares, "top_slices": top}


def _uniform_baseline_and_dominant(aseg, slice_heatmaps: list[dict]) -> tuple[float | None, list[str]]:
    """Return the mean target-voxel percentage of the brain and the two structures with most voxels."""
    if aseg is None:
        return None, []
    tgt_ids = ROI_LABELS["ad_target"]
    fr: list[float] = []
    struct_counts = {name: 0 for name in _STRUCT_IDS}
    for s in slice_heatmaps:
        idx = int(s["index"])
        seg = _slice_from_aseg(aseg, _PLANE_KEYS[idx], _local_index(idx))
        brain = int((seg > 0).sum())
        if brain == 0:
            continue
        fr.append(float(np.isin(seg, tgt_ids).sum()) / brain * 100.0)
        for name, ids in _STRUCT_IDS.items():
            struct_counts[name] += int(np.isin(seg, ids).sum())
    base = float(np.mean(fr)) if fr else None
    dom = [k for k, _ in sorted(struct_counts.items(), key=lambda kv: -kv[1]) if struct_counts[k] > 0][:2]
    return base, dom


def aggregate_volebpg(
    session_id: str,
    canonical_id: str,
    slice_heatmaps: list[dict],
    attention_weights: list[dict],
) -> dict:
    """Return the saliency-concentration fields of the `spatial_xai_anchor` block.

    Without a segmentation or on-target values, `aseg_available` is False and all numeric
    fields are None.
    """
    w_by_idx = {int(a["index"]): float(a.get("percentage", 0.0)) for a in attention_weights}
    have = [(int(s["index"]), s.get("on_target_pct")) for s in slice_heatmaps if s.get("on_target_pct") is not None]

    if not have:
        return {
            "aseg_available": False,
            "on_target_energy_pct": None,
            "baseline_uniform_pct": None,
            "ebpg_concentration_ratio": None,
            "dominant_structures": [],
        }

    wsum = sum(w_by_idx.get(i, 0.0) for i, _ in have) or float(len(have))
    energy = sum(v * (w_by_idx.get(i, 1.0)) for i, v in have) / wsum

    aseg = _load_aseg_for_session(session_id, canonical_id)
    base, dom = _uniform_baseline_and_dominant(aseg, slice_heatmaps)
    ratio = round(energy / base, 2) if base else None

    return {
        "aseg_available": True,
        "on_target_energy_pct": round(float(energy), 2),
        "baseline_uniform_pct": (round(float(base), 2) if base else None),
        "ebpg_concentration_ratio": ratio,
        "dominant_structures": dom,
    }
