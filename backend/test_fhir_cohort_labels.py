"""Cohort-dependent labels in the FHIR bundle.

Sessions of the synthetic BrainWeb demo cohort (ids starting with "BWP") must not present themselves
as OASIS-3 scans, while every other session keeps the OASIS-3 labels unchanged.

Run: `python backend/test_fhir_cohort_labels.py` (or with pytest). No model or server is needed.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fhir_generator import cohort_labels, create_fhir_bundle, validate_fhir_bundle  # noqa: E402

OASIS_SUBJECTS = "https://www.oasis-brains.org/identifiers/subjects"
OASIS_EXPERIMENTS = "https://www.oasis-brains.org/identifiers/experiments"
OASIS_DESCRIPTION = "OASIS-3 Structural Brain MRI (T1w / FreeSurfer Normalized Slices)"
FHIR_ID = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")  # FHIR `id` type, used by ImagingStudy.series.uid


def _bundle(patient_id: str, session_id: str) -> dict:
    return create_fhir_bundle(
        patient_id=patient_id, age=70, gender="female", ai_diagnosis="CN", ai_confidence=0.8,
        probabilities={"CN": 0.8, "MCI": 0.15, "AD": 0.05}, session_id=session_id, slice_count=10,
        model_name="MobileViT25DMultimodalV2-Unified", checkpoint_version="2.0.0",
    )


def _resource(bundle: dict, rtype: str) -> dict:
    return next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == rtype)


def test_oasis_labels_unchanged():
    b = _bundle("OAS30001", "OAS30001_MR_d0129")
    assert _resource(b, "Patient")["identifier"][0]["system"] == OASIS_SUBJECTS
    study = _resource(b, "ImagingStudy")
    assert study["identifier"][0]["system"] == OASIS_EXPERIMENTS
    assert study["description"] == OASIS_DESCRIPTION
    assert study["series"][0]["uid"] == "2.16.840.1.113883.3.oasis3.series.OAS30001.MR.d0129"
    assert _resource(b, "DiagnosticReport")["imagingStudy"][0]["display"] == "OASIS-3 Scan OAS30001_MR_d0129"
    assert _resource(b, "Observation")["derivedFrom"][0]["display"] == "OASIS-3 Scan OAS30001_MR_d0129"


def test_adni_keeps_oasis_labels():
    """Known limitation: ADNI sessions are not relabelled by this change."""
    assert cohort_labels("ADNI_002_S_0729_sc_I40708") is cohort_labels("OAS30001_MR_d0129")


def test_synthetic_labels_have_no_oasis_text():
    b = _bundle("BWP01", "BWP01_MR_d0735")
    assert _resource(b, "Patient")["identifier"][0]["system"] == "urn:alzai:synthetic:subjects"
    study = _resource(b, "ImagingStudy")
    assert study["identifier"][0]["system"] == "urn:alzai:synthetic:experiments"
    assert study["description"].startswith("Synthetic BrainWeb phantom")
    uid = study["series"][0]["uid"]
    assert uid == "2.16.840.1.113883.3.synthetic.series.BWP01.MR.d0735" and FHIR_ID.match(uid)
    assert _resource(b, "DiagnosticReport")["imagingStudy"][0]["display"] == "Synthetic BrainWeb phantom BWP01_MR_d0735"
    assert _resource(b, "Observation")["derivedFrom"][0]["display"] == "Synthetic BrainWeb phantom BWP01_MR_d0735"
    assert "oasis" not in json.dumps(b).lower()


def test_cohort_is_case_insensitive():
    assert cohort_labels("bwp02_MR_d0371") is cohort_labels("BWP02_MR_d0371")
    assert cohort_labels("BWP02_MR_d0371") is not cohort_labels("OAS30001_MR_d0129")


def test_bundles_still_validate():
    for pid, sid in (("OAS30001", "OAS30001_MR_d0129"), ("BWP01", "BWP01_MR_d0735")):
        report = validate_fhir_bundle(_bundle(pid, sid))
        assert report["valid"], report["errors"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
