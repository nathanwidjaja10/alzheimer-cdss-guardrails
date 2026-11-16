#!/usr/bin/env python3
"""Generate the static assets of the "Anatomical ROI Reference" viewer in the CDSS.

Writes to alzheimer-xai-cdss/public/demo/:
  - brain_cn.mgz, brain_ad.mgz    copies of the dummy brain.mgz volumes (BrainWeb phantoms)
  - roi_cn.nii.gz, roi_ad.nii.gz  aseg.mgz reduced to the Alzheimer target structures,
                                  relabelled 1..5 (all other voxels 0)
  - roi_demo_meta.json            dummy session ids and covariates, reference structure volumes

The images and covariates are synthetic (dummy_data/, see dummy_data/README.md). The two panels are named "cn" and
"ad" after the reference volumes they illustrate. Both phantoms have normal anatomy, so the structure volumes shown
in the pop-up are NOT measured on these images: they are typical values of one cognitively normal and one AD
dementia OASIS-3 sample (REFERENCE_VOLUMES_ML below), and the pop-up says so. The volumes measured on the phantoms
are written to the metadata as "image_measured_ml" for transparency only.

Label remap (the AD target structures used in the anatomical XAI benchmark):
  1 = Hippocampus            (aseg 17, 53)
  2 = Amygdala               (aseg 18, 54)
  3 = Lateral ventricle      (aseg  4, 43)   body / atrium
  4 = Temporal horn          (aseg  5, 44)   inferior lateral ventricle
  5 = Third ventricle        (aseg 14)

Run from the repository root after changing the sample sessions:
    python alzheimer-xai-cdss/generate_roi_demo.py
"""
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FS_DIR = REPO / "dummy_data" / "freesurfer_data"
OUT_DIR = Path(__file__).resolve().parent / "public" / "demo"

# Dummy sessions (covariates from dummy_data/clinical_manifest.csv, made up)
SAMPLES = {
    "cn": {"session": "BWP01_MR_d0000", "cdr": 0, "mmse": 30, "age": 68,
           "diagnosis": "Cognitively normal"},
    "ad": {"session": "BWP04_MR_d0728", "cdr": 2, "mmse": 14, "age": 81,
           "diagnosis": "AD Dementia"},
}

# Reference structure volumes in mL (both hemispheres) of one cognitively normal and one AD dementia OASIS-3 sample
# (aseg voxel counts at 1 mm^3). Shown in the pop-up as "from OASIS-3 samples, not measured on this image".
REFERENCE_VOLUMES_ML = {
    "cn": {"1": 7.64, "2": 2.85, "3": 26.0, "4": 1.24, "5": 1.15},
    "ad": {"1": 2.73, "2": 1.53, "3": 73.75, "4": 6.84, "5": 1.86},
}

REMAP = {
    1: (17, 53),   # hippocampus
    2: (18, 54),   # amygdala
    3: (4, 43),    # lateral ventricle (body / atrium)
    4: (5, 44),    # temporal horn (inferior lateral ventricle)
    5: (14,),      # third ventricle
}


def build(session: str, tag: str) -> dict:
    """Write the ROI label volume and brain volume for one dummy session; return the volumes measured on the phantom, in mL."""
    src = FS_DIR / session
    aseg_path = src / "aseg.mgz"
    brain_path = src / "brain.mgz"
    if not aseg_path.exists() or not brain_path.exists():
        raise SystemExit(f"missing brain.mgz/aseg.mgz for {session}")

    aseg = nib.load(str(aseg_path))
    data = np.asarray(aseg.dataobj)
    roi = np.zeros(data.shape, dtype=np.uint8)
    vox_mm3 = abs(np.linalg.det(aseg.affine[:3, :3]))

    # Relabel the target structures and measure their volumes on the phantom
    vols = {}
    for new_label, old_labels in REMAP.items():
        mask = np.isin(data, list(old_labels))
        roi[mask] = new_label
        vols[new_label] = round(float(mask.sum() * vox_mm3 / 1000.0), 2)  # mL

    out_roi = OUT_DIR / f"roi_{tag}.nii.gz"
    nib.save(nib.Nifti1Image(roi, aseg.affine), str(out_roi))
    shutil.copyfile(brain_path, OUT_DIR / f"brain_{tag}.mgz")

    print(f"[{tag}] {session}  hippo={vols[1]} mL  amyg={vols[2]} mL  "
          f"latVent={vols[3]} mL  tempHorn={vols[4]} mL  v3={vols[5]} mL")
    return {"image_measured_ml": vols}


def main() -> None:
    """Generate the assets for every sample session."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"label_map": {
        "1": "Hippocampus", "2": "Amygdala", "3": "Lateral ventricle",
        "4": "Temporal horn", "5": "Third ventricle"}}
    meta["volumes_source"] = "OASIS-3 samples (CN and AD), not measured on the dummy images"
    for tag, info in SAMPLES.items():
        meta[tag] = {**info, "synthetic": True, "volumes_ml": REFERENCE_VOLUMES_ML[tag], **build(info["session"], tag)}
    (OUT_DIR / "roi_demo_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
