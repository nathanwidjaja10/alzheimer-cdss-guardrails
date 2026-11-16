#!/usr/bin/env python3
"""Build the synthetic demo sessions of `dummy_data/` from the BrainWeb Simulated Brain Database.

The sessions are simulated normal-brain T1 images, grouped into dummy patients with made-up covariates
(`dummy_data/clinical_manifest.csv`). They are not patient data and carry no clinical meaning; they let the
CDSS run end to end (including the longitudinal timeline) without OASIS-3 or ADNI. See `dummy_data/README.md`.

Stages (each skips work that already exists; `all` runs them in order):

    download   fetch the simulated T1 of each BrainWeb subject into dummy_data/raw_brainweb/ (not tracked)
    segment    FastSurfer segmentation (aseg + brain mask) -> dummy_data/freesurfer_data/<session>/{brain,aseg}.mgz
    slices     10 multi-plane slices per session, cut exactly as in the training notebook -> dummy_data/slice_cache/
    metadata   dummy_data/clinical_metadata.json (the format the backend reads) from clinical_manifest.csv
    facts      deterministic fact context of each session -> dummy_data/gold_facts/ (needs the model checkpoint)

    python tools/make_brainweb_demo.py all
    python tools/make_brainweb_demo.py segment --device cpu
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DD = REPO / "dummy_data"
RAW = DD / "raw_brainweb"
WORK = DD / "_work"
FASTSURFER = REPO / "tools" / "FastSurfer"
MANIFEST = DD / "clinical_manifest.csv"
SOURCES = DD / "brainweb_sources.csv"

BRAINWEB_URL = "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1"

# Slice bands and counts: identical to the training notebook (cell 4) and backend/inference_engine.py
N_AXIAL, N_CORONAL, N_SAGITTAL = 5, 3, 2
AXIAL_PCT, CORONAL_PCT, SAGITTAL_PCT = (0.35, 0.65), (0.35, 0.65), (0.40, 0.60)


def read_manifest() -> list[dict]:
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- download
def download(rows: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for subj in sorted({r["brainweb_subject"] for r in rows}):
        dest = RAW / f"subject{subj}_t1w.mnc.gz"
        if not dest.exists() or not _gzip_ok(dest):
            form = urllib.parse.urlencode({
                "do_download_alias": f"subject{subj}_t1w", "format_value": "minc", "zip_value": "gnuzip",
                "who_name": "", "who_institution": "", "who_email": "",  # optional fields, left blank
                "download_for_real": "[Start download!]",
            }).encode()
            print(f"  downloading BrainWeb subject {subj} ...")
            with urllib.request.urlopen(urllib.request.Request(BRAINWEB_URL, data=form), timeout=300) as r, open(dest, "wb") as o:
                shutil.copyfileobj(r, o)
            if not _gzip_ok(dest):
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"download of subject {subj} failed")
            time.sleep(2)
        out_rows.append({"brainweb_subject": subj, "alias": f"subject{subj}_t1w", "file": dest.name,
                         "bytes": dest.stat().st_size, "sha256": _sha256(dest)})
    with open(SOURCES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"  {len(out_rows)} subjects; checksums -> {SOURCES.relative_to(REPO)}")


def _gzip_ok(p: Path) -> bool:
    try:
        with gzip.open(p, "rb") as f:
            while f.read(1 << 22):
                pass
        return True
    except (OSError, EOFError):
        return False


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- segment
def segment(rows: list[dict], device: str) -> None:
    import nibabel as nib

    for r in rows:
        sid, subj = r["session_id"], r["brainweb_subject"]
        fs_dir = DD / "freesurfer_data" / sid
        if (fs_dir / "brain.mgz").exists() and (fs_dir / "aseg.mgz").exists():
            print(f"  {sid}: segmentation exists, skipped")
            continue
        work = WORK / sid
        (work / "mri").mkdir(parents=True, exist_ok=True)
        (work / "scripts").mkdir(exist_ok=True)

        # BrainWeb MINC -> float NIfTI (FastSurfer conforms it to 256^3, 1 mm, LIA, uint8)
        t1 = work / "T1.nii.gz"
        if not t1.exists():
            mnc = work / "T1.mnc"
            with gzip.open(RAW / f"subject{subj}_t1w.mnc.gz", "rb") as f, open(mnc, "wb") as o:
                shutil.copyfileobj(f, o)
            im = nib.load(str(mnc))
            out = nib.Nifti1Image(np.asarray(im.dataobj, dtype=np.float32), im.affine)
            out.header.set_data_dtype(np.float32)
            nib.save(out, str(t1))
            mnc.unlink()

        mri = work / "mri"
        cmd = [sys.executable, str(FASTSURFER / "FastSurferCNN" / "run_prediction.py"),
               "--t1", str(t1), "--sid", sid, "--sd", str(WORK),
               "--asegdkt_segfile", str(mri / "aparc.DKTatlas+aseg.deep.mgz"),
               "--conformed_name", str(mri / "orig.mgz"), "--brainmask_name", str(mri / "mask.mgz"),
               "--aseg_name", str(mri / "aseg.auto_noCC.mgz"), "--seg_log", str(work / "scripts" / "deep-seg.log"),
               "--batch_size", "1", "--device", device, "--viewagg_device", device, "--threads", "4"]
        print(f"  {sid}: FastSurfer segmentation ({device}) ...")
        t0 = time.perf_counter()
        env = dict(os.environ, PYTHONPATH=str(FASTSURFER))
        proc = subprocess.run(cmd, cwd=FASTSURFER, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
            raise RuntimeError(f"FastSurfer failed for {sid} (exit {proc.returncode})")

        # brain.mgz = conformed T1 inside the brain mask; aseg.mgz = reduced aseg (same grid)
        orig = nib.load(str(mri / "orig.mgz"))
        mask = np.asarray(nib.load(str(mri / "mask.mgz")).dataobj) > 0
        brain = np.where(mask, np.asarray(orig.dataobj), 0).astype(np.uint8)
        fs_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.MGHImage(brain, orig.affine), str(fs_dir / "brain.mgz"))
        aseg = nib.load(str(mri / "aseg.auto_noCC.mgz"))
        nib.save(nib.MGHImage(np.asarray(aseg.dataobj).astype(np.uint8), aseg.affine), str(fs_dir / "aseg.mgz"))
        print(f"  {sid}: done in {time.perf_counter() - t0:.0f} s")


# --------------------------------------------------------------------------- slices
def normalize_volume_3d(data: np.ndarray) -> np.ndarray:
    """Robust 1st-99th percentile intensity normalisation over non-zero voxels (as in the notebook)."""
    nz = data[data > 0]
    v0, v1 = float(np.percentile(nz, 1)), float(np.percentile(nz, 99))
    if v1 - v0 < 1e-6:
        v1 = v0 + 1.0
    return np.clip((data - v0) / (v1 - v0) * 255.0, 0, 255).astype(np.uint8)


def extract_slices(brain: np.ndarray):
    """5 + 3 + 2 slices of the LIA volume: slice_*.png (anatomically coronal), coronal_*.png (axial), sagittal_*.png."""
    d = normalize_volume_3d(brain.astype(np.float32))
    ax = [np.rot90(d[:, :, z], 1) for z in np.linspace(int(d.shape[2] * AXIAL_PCT[0]), int(d.shape[2] * AXIAL_PCT[1]) - 1, N_AXIAL, dtype=int)]
    co = [np.rot90(d[:, y, :], 1) for y in np.linspace(int(d.shape[1] * CORONAL_PCT[0]), int(d.shape[1] * CORONAL_PCT[1]) - 1, N_CORONAL, dtype=int)]
    sa = [np.rot90(d[x, :, :], 1) for x in np.linspace(int(d.shape[0] * SAGITTAL_PCT[0]), int(d.shape[0] * SAGITTAL_PCT[1]) - 1, N_SAGITTAL, dtype=int)]
    return ax, co, sa


def slices(rows: list[dict]) -> None:
    import nibabel as nib
    from PIL import Image

    for r in rows:
        sid = r["session_id"]
        out = DD / "slice_cache" / sid
        if len(list(out.glob("*.png"))) == N_AXIAL + N_CORONAL + N_SAGITTAL:
            print(f"  {sid}: slices exist, skipped")
            continue
        out.mkdir(parents=True, exist_ok=True)
        brain = np.asarray(nib.load(str(DD / "freesurfer_data" / sid / "brain.mgz")).dataobj)
        ax, co, sa = extract_slices(brain)
        for i, s in enumerate(ax):
            Image.fromarray(s, "L").save(out / f"slice_{i:03d}.png")
        for i, s in enumerate(co):
            Image.fromarray(s, "L").save(out / f"coronal_{i:03d}.png")
        for i, s in enumerate(sa):
            Image.fromarray(s, "L").save(out / f"sagittal_{i:03d}.png")
        print(f"  {sid}: 10 slices")


# --------------------------------------------------------------------------- metadata
def metadata(rows: list[dict]) -> None:
    meta = {}
    for r in rows:
        meta[r["session_id"]] = {
            "subject_id": r["subject_id"], "gender": r["gender"], "education": r["education"], "apoe": r["apoe"],
            "mr_days": int(r["days_from_entry"]), "clinical_match_diff_days": 0, "age_at_scan": r["age_at_scan"],
            "cdr": r["cdr"], "mmse": r["mmse"], "diagnosis": r["diagnosis"], "synthetic": True,
        }
    p = DD / "clinical_metadata.json"
    p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {len(meta)} sessions -> {p.relative_to(REPO)}")


# --------------------------------------------------------------------------- facts
def facts(rows: list[dict]) -> None:
    sys.path.insert(0, str(REPO))
    from backend.inference_engine import predict_session
    from llm_xai.backend.llm_engine import build_fact_context

    out_dir = DD / "gold_facts"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for r in rows:
        sid = r["session_id"]
        t0 = time.perf_counter()
        ai = predict_session(sid, xai_method="hirescam")
        fc = build_fact_context(ai, sid).model_dump()
        (out_dir / f"{sid}.json").write_text(json.dumps(fc, indent=2, default=str))
        pa, sx = fc["prediction_anchor"], fc["spatial_xai_anchor"]
        index.append({"session_id": sid, "argmax_class": pa["argmax_class"], "ood_status": pa["ood_status"],
                      "aseg_available": sx["aseg_available"], "primary_plane": sx["primary_plane"],
                      "dominant_structures": ";".join(sx.get("dominant_structures") or []), "fact_digest": fc["fact_digest"]})
        print(f"  {sid}: {pa['argmax_class']} / {pa['ood_status']} ({time.perf_counter() - t0:.1f} s)")
    with open(DD / "gold_facts_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"  index -> dummy_data/gold_facts_index.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "segment", "slices", "metadata", "facts", "all"])
    ap.add_argument("--device", default="auto", help="FastSurfer device: auto, cpu, mps, cuda")
    args = ap.parse_args()
    rows = read_manifest()
    stages = ["download", "segment", "slices", "metadata", "facts"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"[{st}]")
        if st == "download":
            download(rows)
        elif st == "segment":
            segment(rows, args.device)
        elif st == "slices":
            slices(rows)
        elif st == "metadata":
            metadata(rows)
        elif st == "facts":
            facts(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
