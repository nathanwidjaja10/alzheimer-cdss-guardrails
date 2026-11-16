# dummy_data: synthetic demo sessions

Twelve simulated brain MRI sessions that let anyone run the CDSS end to end, including the longitudinal
timeline, **without OASIS-3 or ADNI**. This repository does not redistribute those datasets (their data use
terms do not allow it), so the demo data are synthetic.

> **These are not patient data and carry no clinical meaning.**
> The images are simulated normal brains, the covariates are made up, and whatever stage the model assigns is a
> software smoke test, not a result. Nothing here is part of the experiments in the paper.

## What is in the folder

| Path | Content | Tracked |
|---|---|---|
| `clinical_manifest.csv` | Dummy clinical manifest: 4 dummy patients, 3 sessions each, with made-up age, sex, education, APOE, MMSE, CDR, and the BrainWeb subject used for each session | yes |
| `clinical_metadata.json` | Same content in the format the backend reads (generated from the CSV) | yes |
| `freesurfer_data/<session>/brain.mgz`, `aseg.mgz` | Skull-stripped T1 and FastSurfer `aseg`, 256³, 1 mm, LIA (same layout as the OASIS-3 folders) | yes |
| `slice_cache/<session>/*.png` | The 10 multi-plane slices the model takes (5 + 3 + 2), cut exactly as in the training notebook | yes |
| `gold_facts/<session>.json`, `gold_facts_index.csv` | Deterministic fact context of each session (input of the LLM layer) | yes |
| `brainweb_sources.csv` | BrainWeb subject, file name, size, and SHA-256 of each downloaded original | yes |
| `raw_brainweb/` | The downloaded BrainWeb originals (`.mnc.gz`, about 17 MB each) | no, rebuilt by the script |
| `_work/` | FastSurfer working files | no |

## The dummy patients

| Patient | Story | Sessions (day: MMSE, CDR) |
|---|---|---|
| `BWP01` | stable normal | 0: 30, 0 · 364: 29, 0 · 735: 30, 0 |
| `BWP02` | normal to mild decline | 0: 29, 0 · 371: 27, 0.5 · 742: 25, 0.5 |
| `BWP03` | mild decline to dementia | 0: 26, 0.5 · 378: 23, 1 · 749: 19, 1 |
| `BWP04` | progressing dementia | 0: 21, 1 · 357: 18, 1 · 728: 14, 2 |

Every session is a **different** BrainWeb phantom (no phantom has real follow-up scans), so the images do not
show disease progression. The longitudinal view demonstrates the interface only. In the CDSS they appear under
the cohort filter "Synthetic".

## How it was built

```bash
python tools/make_brainweb_demo.py all        # download, segment, slices, metadata, facts
python tools/make_brainweb_demo.py segment --device cpu    # a single stage, on CPU
```

1. `download`: the simulated T1 of each BrainWeb subject (the optional name, institution, and e-mail fields of the
   BrainWeb form are left blank).
2. `segment`: [FastSurfer](https://github.com/Deep-MI/FastSurfer) segmentation only (`tools/FastSurfer`),
   which gives the brain mask and `aseg`. `brain.mgz` is the conformed T1 inside that mask.
3. `slices`, `metadata`, `facts`: the slice cache, the backend metadata file, and the fact contexts (the last
   needs the model checkpoint).

## What the CDSS shows for these sessions

Checked with the backend and frontend of this repository (`/api/analyze`, HiResCAM, then the web interface):

| Patient | Predicted stage (argmax) | P(CN) | Out-of-domain gate |
|---|---|---|---|
| `BWP01` (3 sessions) | CN, CN, CN | 0.83, 0.78, 0.70 | within range, within range, **outside** |
| `BWP02` (3 sessions) | CN, CN, CN | 0.74, 0.77, 0.75 | within range |
| `BWP03` (3 sessions) | CN, CN, CN | 0.79, 0.75, 0.68 (P(AD) up to 0.25) | within range |
| `BWP04` (3 sessions) | CN, CN, CN | 0.79, 0.76, 0.82 | within range |

- All 12 sessions are staged CN, which is expected for simulated normal brains. The dummy CDR and MMSE do not
  feed the image-only decision; they are shown as context and written into the FHIR bundle.
- The out-of-domain badge appears for one session only (`BWP01_MR_d0735`, squared Mahalanobis distance 999.7
  against a threshold of 997.5), so these phantoms mostly lie inside the validated range.
- `aseg` structures come from FastSurfer on a phantom (hippocampus 5.5-7.3 mL, lateral ventricles 18-40 mL), so the
  per-structure saliency numbers and the dominant structure are computed but meaningless.
- The FHIR bundle validates, and the guardrailed LLM narrative and Q&A work as for any session, describing
  made-up inputs.

## Source of the images and how to cite

The images come from the **BrainWeb: Simulated Brain Database** (McConnell Brain Imaging Centre, McGill University,
<https://brainweb.bic.mni.mcgill.ca/>): the 20 normal anatomical models, simulated T1 (spoiled FLASH, TR 22 ms,
TE 9.2 ms, flip angle 30°, 1 mm isotropic). The site asks that its references be cited in any published work that
uses the data, and states no redistribution license, so only these derived files are kept here and the originals are
fetched from BrainWeb by the script:

- C.A. Cocosco, V. Kollokian, R.K.-S. Kwan, A.C. Evans, "BrainWeb: Online Interface to a 3D MRI Simulated Brain
  Database," NeuroImage 5(4), S425, 1997.
- R.K.-S. Kwan, A.C. Evans, G.B. Pike, "MRI simulation-based evaluation of image-processing and classification
  methods," IEEE Trans. Med. Imaging 18(11), 1085-1097, 1999.
- D.L. Collins et al., "Design and construction of a realistic digital brain phantom," IEEE Trans. Med. Imaging
  17(3), 463-468, 1998.
- B. Aubert-Broche, A.C. Evans, D.L. Collins, "A new improved version of the realistic digital brain phantom,"
  NeuroImage, 2006 (the 20 anatomical models; check the final citation details).

Segmentation: L. Henschel et al., "FastSurfer - a fast and accurate deep learning based neuroimaging pipeline,"
NeuroImage 219, 117012, 2020.
