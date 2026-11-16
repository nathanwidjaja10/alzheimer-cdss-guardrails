"""Embed a verified narrative into a FHIR bundle.

Builds the bundle with `backend.fhir_generator.create_fhir_bundle`, then replaces
`DiagnosticReport.conclusion` with the narrative and attaches an audit tag (fact digest and
guardrail status). Called only for narratives that passed the guardrail or are the deterministic
fallback.
"""
from __future__ import annotations

import re
from typing import Any

from .guardrails import flatten_narrative

# Sanitization limits for narrative text. The schema validator accepts empty text, control
# characters and arbitrarily long text, so these are enforced here (see
# notebooks/fhir_bundle_validation.ipynb).
_CONTROL_CHAR_RX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NARRATIVE_MAX_LEN = 20_000
_NARRATIVE_EMPTY_PLACEHOLDER = "[content unavailable]"


def sanitize_narrative(narrative: dict) -> dict:
    """Clean each narrative field before it is placed in the FHIR bundle.

    Coerces values to str, strips control characters (except newline and tab), truncates
    over-long text, and replaces empty fields with a placeholder.
    """
    cleaned: dict[str, str] = {}
    for key, value in narrative.items():
        text = "" if value is None else str(value)
        text = _CONTROL_CHAR_RX.sub("", text).strip()
        if len(text) > _NARRATIVE_MAX_LEN:
            text = text[:_NARRATIVE_MAX_LEN].rsplit(" ", 1)[0] + " [truncated]"
        cleaned[key] = text or _NARRATIVE_EMPTY_PLACEHOLDER
    return cleaned

try:
    from backend.fhir_generator import create_fhir_bundle
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.fhir_generator import create_fhir_bundle


def _find_resource(bundle: dict, rtype: str) -> dict | None:
    """Return the first resource of the given type in the bundle, or None."""
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") == rtype:
            return res
    return None


def build_bundle_with_narrative(
    *,
    narrative: dict,
    provenance: dict,
    ai_result: dict,
    session_id: str,
    clinical: dict | None = None,
) -> dict:
    """Build a FHIR R4 bundle and inject the narrative and audit tag into its DiagnosticReport."""
    clinical = clinical or {}
    bundle: dict[str, Any] = create_fhir_bundle(
        patient_id=clinical.get("subject_id") or session_id.split("_")[0],
        age=clinical.get("age", "Unknown"),
        gender=clinical.get("gender", "Unknown"),
        ai_diagnosis=ai_result["diagnosis"],
        ai_confidence=ai_result["confidence"],
        probabilities=ai_result.get("probabilities"),
        session_id=session_id,
        slice_count=ai_result.get("slice_count"),
        gradcam_base64=ai_result.get("gradcam_base64"),
        model_name=ai_result.get("model_info", {}).get("name", "MobileViT25DMultimodalV2-Unified"),
        execution_latency_ms=ai_result.get("latency_ms"),
        mmse=clinical.get("mmse"),
        cdr=clinical.get("cdr"),
        apoe=clinical.get("apoe"),
    )

    dr = _find_resource(bundle, "DiagnosticReport")
    if dr is not None:
        dr["conclusion"] = flatten_narrative(sanitize_narrative(narrative))
        digest = str(provenance.get("fact_digest", ""))
        tag = {
            "system": "urn:alzai:llm-xai",
            "code": (digest.split(":")[-1][:20] or "unknown"),
            "display": (
                f"guardrail={provenance.get('guardrail_status', 'n/a')}; "
                f"source={provenance.get('narrative_source', 'n/a')}; "
                f"model={provenance.get('model', 'n/a')}"
            ),
        }
        dr.setdefault("meta", {}).setdefault("tag", []).append(tag)

    return bundle
