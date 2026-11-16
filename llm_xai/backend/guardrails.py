"""Deterministic algorithmic guardrails (DAG): a non-LLM verifier for generated narratives and answers.

Narrative auditors:
  1. audit_numeric           every number must match a value in the fact context
  2. audit_anatomy           tier-C structures are fabrications; tier-B terms must not carry a number
  3. audit_contradiction     differential class equals the argmax; out-of-domain consistency;
                             APOE caveats (no diagnostic use, no denial of an E4 allele);
                             no absolute confidence percentages
  4. audit_structure_binding slice-name/percentage pairs and dominant-structure claims must
                             match their fact fields

`run_dag()` runs the four auditors and computes the numeric consistency rate (NCR).
`render_fallback()` builds a template narrative that always passes the DAG.
Q&A mode adds further auditors (see `run_qa_dag`).
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import GUARDRAIL_VERSION, QA_GUARDRAIL_VERSION
from .anatomical_whitelist import scan
from .llm_engine import MEASURED_STRUCTURES, FactContext, NarrativeOut, resolve_fact_path

if TYPE_CHECKING:  # qa_engine imports from this module, so it is imported only for type checking
    from .qa_engine import AnswerOut, QASession, QATurn

# Numbers that may always appear (slice count, resolution, feature width, session count)
STRUCTURAL_INTS = set(range(0, 11)) | {224, 232, 640}

_NUM_RX = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*(%?)")


# ===========================================================================
# Result structures
# ===========================================================================
@dataclass
class Violation:
    type: str
    detail: str
    found: str = ""
    expected: str = ""


@dataclass
class AuditResult:
    name: str
    passed: bool
    violations: list[Violation] = field(default_factory=list)
    checked: int = 0
    valid: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "checked": self.checked,
            "valid": self.valid,
            "violations": [v.__dict__ for v in self.violations],
        }


@dataclass
class GuardrailReport:
    passed: bool
    source: str  # "llm" | "llm_regenerated" | "deterministic_fallback"
    audits: list[AuditResult]
    ncr: float
    version: str = GUARDRAIL_VERSION

    def all_violations(self) -> list[Violation]:
        return [v for a in self.audits for v in a.violations]

    def summary(self) -> str:
        vs = self.all_violations()
        if not vs:
            return "No violations."
        return "Fix the following without adding any new claim:\n" + "\n".join(
            f"- [{v.type}] {v.detail}"
            + (f" (found: '{v.found}')" if v.found else "")
            + (f" (expected: '{v.expected}')" if v.expected else "")
            for v in vs
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "source": self.source,
            "ncr": self.ncr,
            "version": self.version,
            "audits": [a.to_dict() for a in self.audits],
            "violation_types": sorted({v.type for v in self.all_violations()}),
        }


# ===========================================================================
# Numeric values permitted by the fact context
# ===========================================================================
def allowed_numbers(facts: FactContext) -> list[tuple[float, float, str]]:
    """Return (value, tolerance, label) for every number the fact context permits in a narrative."""
    out: list[tuple[float, float, str]] = []
    pa, sx, cc = facts.prediction_anchor, facts.spatial_xai_anchor, facts.clinical_covariates

    for k, p in pa.relative_probabilities.items():
        out.append((round(float(p), 4), 0.02, f"prob[{k}]"))
        out.append((round(float(p) * 100, 1), 0.7, f"prob%[{k}]"))

    for plane, v in (sx.plane_attention_share_pct or {}).items():
        out.append((round(float(v), 1), 0.7, f"plane_share[{plane}]"))

    for t in sx.top_slices:
        out.append((round(float(t.attention_share_pct), 1), 0.7, f"slice_share[{t.name}]"))
        if t.on_target_pct is not None:
            out.append((round(float(t.on_target_pct), 2), 0.7, f"slice_on_target[{t.name}]"))

    for val, lbl in (
        (sx.on_target_energy_pct, "on_target_energy_pct"),
        (sx.baseline_uniform_pct, "baseline_uniform_pct"),
    ):
        if val is not None:
            out.append((round(float(val), 2), 0.7, lbl))
    if sx.ebpg_concentration_ratio is not None:
        out.append((round(float(sx.ebpg_concentration_ratio), 2), 0.15, "ebpg_ratio"))

    if cc.age is not None:
        out.append((float(cc.age), 1.0, "age"))
    if cc.mmse_score is not None:
        out.append((float(cc.mmse_score), 0.6, "mmse"))
    if cc.cdr_score is not None:
        out.append((float(cc.cdr_score), 0.05, "cdr"))

    for val, lbl in (
        (pa.sq_mahalanobis_distance, "sq_mahalanobis"),
        (pa.ood_threshold, "ood_threshold"),
    ):
        if val is not None:
            out.append((float(val), max(5.0, 0.05 * abs(float(val))), lbl))

    return out


# ===========================================================================
# Auditor 1 — numeric
# ===========================================================================
def audit_numeric(text: str, facts: FactContext) -> AuditResult:
    """Flag numbers in the text that match no value in the fact context (within tolerance)."""
    allowed = allowed_numbers(facts)
    checked = valid = 0
    viol: list[Violation] = []
    for m in _NUM_RX.finditer(text):
        raw = m.group(0).strip()
        num = float(m.group(1))
        is_pct = m.group(2) == "%"
        if not is_pct and num.is_integer() and int(num) in STRUCTURAL_INTS:
            continue
        checked += 1
        if any(abs(num - val) <= tol for val, tol, _ in allowed):
            valid += 1
        else:
            viol.append(Violation("ungrounded_number", f"the number '{raw}' does not appear in the Fact Context", found=raw))
    return AuditResult("numeric", not viol, viol, checked, valid)


# ===========================================================================
# Auditor 2 — anatomy
# ===========================================================================
def audit_anatomy(text: str) -> AuditResult:
    """Flag structures outside the measured set (tier C) and context terms given a number (tier B)."""
    viol: list[Violation] = []
    for h in scan(text):
        if h["tier"] == "C":
            viol.append(Violation("fabricated_structure",
                                  f"structure outside the model's measured set: '{h['term']}'", found=h["term"]))
        elif h["tier"] == "B":
            window = text[max(0, h["start"] - 40): h["end"] + 40]
            if re.search(r"\d", window):
                viol.append(Violation("quantified_context_term",
                                      f"broad context term '{h['term']}' was given a quantitative number", found=h["term"]))
    return AuditResult("anatomy", not viol, viol)


# ===========================================================================
# Auditor 3 — contradiction
# ===========================================================================
_OOD_OUT_RX = re.compile(r"out[- ]of[- ]domain|outside[^.]{0,25}validat|beyond[^.]{0,25}domain|not[^.]{0,15}validat", re.I)
_OOD_IN_RX = re.compile(r"within[^.]{0,25}validat|in[- ]domain|inside[^.]{0,25}validat", re.I)
_APOE_AD_RX = re.compile(
    # APOE used as evidence for AD. The gaps use `.` (not `[^.]`), so a match may span sentences
    # within one field but stops at a field boundary (fields are joined with newlines).
    r"apoe.{0,160}(support|consistent with|rais\w+|increas\w+|elevat\w+|confirm\w+|strengthen\w+|diagnos\w+|indicat\w+).{0,40}(ad\b|alzheimer|risk)"
    r"|(support|consistent with|rais\w+|increas\w+|strengthen\w+|diagnos\w+|indicat\w+).{0,40}(ad\b|alzheimer).{0,120}apoe",
    re.I,
)
_ABS_CONF_RX = re.compile(r"(confiden\w+|certain\w+)[^.\d]{0,15}\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*%[^.\d]{0,15}(confiden\w+|certain\w+)", re.I)
_APOE_NO_E4_RX = re.compile(
    # Denial of an APOE E4 allele. The first alternative requires "no" directly before "apoe" so
    # that epistemic phrasing ("no indication that the APOE4 allele provides evidence ...") is
    # not matched.
    r"no\s+apoe.{0,20}(?:4|e4|allele)"
    r"|apoe.{0,20}(?:4|e4).{0,20}(?:absent|not present|not detected|does not carry|no evidence of)"
    r"|absence of (?:the )?apoe.{0,10}(?:4|e4)",
    re.I,
)
_CONCENTRATION_VERB_RX = re.compile(
    r"(?:most |primarily |predominantly )?(?:being\s+)?"
    r"(?:concentrat\w+|focus(?:ed|ing)?|emphasi[sz]\w+)\s+(?:in|on)\s+(?:the\s+)?",
    re.I,
)


def audit_contradiction(narr: NarrativeOut, facts: FactContext) -> AuditResult:
    """Flag narrative statements that contradict the fact context (class, domain status, APOE, confidence)."""
    viol: list[Violation] = []
    pa = facts.prediction_anchor

    if narr.differential_conclusion != pa.argmax_class:
        viol.append(Violation("diagnosis_mismatch",
                              "differential_conclusion != argmax_class",
                              found=narr.differential_conclusion, expected=pa.argmax_class))

    text = narr.as_text()
    if _OOD_OUT_RX.search(text) and pa.ood_status != "outside_validated_domain":
        viol.append(Violation("ood_contradiction",
                              f"narrative implies outside the validated domain, but ood_status={pa.ood_status}"))
    if _OOD_IN_RX.search(text) and pa.ood_status == "outside_validated_domain":
        viol.append(Violation("ood_contradiction",
                              "narrative implies within the validated domain, but ood_status=outside_validated_domain"))

    if facts.clinical_covariates.apoe_available and _APOE_AD_RX.search(text):
        viol.append(Violation("covariate_caveat_breach", "APOE4 is used as supporting evidence for AD"))

    cc = facts.clinical_covariates
    has_e4 = bool(cc.apoe_available and cc.apoe_genotype and "4" in cc.apoe_genotype)
    if has_e4 and _APOE_NO_E4_RX.search(text):
        viol.append(Violation("apoe_negation_contradiction",
                              f"narrative denies APOE4 presence but genotype={cc.apoe_genotype} contains e4"))

    if _ABS_CONF_RX.search(text):
        viol.append(Violation("absolute_confidence_claim",
                              "narrative claims an absolute confidence percentage (must be a relative ranking)"))

    return AuditResult("contradiction", not viol, viol)


def audit_structure_binding(text: str, facts: FactContext) -> AuditResult:
    """Check that slice labels are paired with their own attention share and that saliency claims name a dominant structure.

    (a) Each "<plane> <index> ... <number>" pair must carry that slice's attention share;
        a number belonging to another slice is reported as swapped.
    (b) A claim that saliency concentrates in a structure must name one of
        `spatial_xai_anchor.dominant_structures`.

    Coverage is deliberately narrow to favour precision.
    """
    sx = facts.spatial_xai_anchor
    viol: list[Violation] = []
    checked = valid = 0

    slice_labels: dict[str, float] = {
        f"{t.name} {t.index}": t.attention_share_pct for t in sx.top_slices if t.name
    }
    for label, num in _extract_label_value_pairs(text, list(slice_labels.keys())):
        checked += 1
        true_val = slice_labels[label]
        if _value_matches(num, true_val):
            valid += 1
            continue
        swapped_with = next(
            (other for other, ov in slice_labels.items() if other != label and _value_matches(num, ov)),
            None,
        )
        detail = f"'{label}' paired with {num}, but its true attention share is {true_val}"
        if swapped_with:
            detail += f" (this number belongs to '{swapped_with}' -- looks swapped)"
        viol.append(Violation("slice_binding_mismatch", detail, found=f"{label}={num}", expected=f"{label}={true_val}"))

    if sx.dominant_structures:
        dominant_lower = {d.lower() for d in sx.dominant_structures}
        for m in _CONCENTRATION_VERB_RX.finditer(text):
            window = text[m.end(): m.end() + 40]
            for term in MEASURED_STRUCTURES:
                if window.lower().startswith(term.lower()):
                    checked += 1
                    if term.lower() in dominant_lower:
                        valid += 1
                    else:
                        viol.append(Violation(
                            "dominant_structure_mismatch",
                            f"narrative claims saliency concentrates in '{term}', but the model's actual "
                            f"dominant structure(s) for this session are {sx.dominant_structures}",
                            found=term, expected="/".join(sx.dominant_structures),
                        ))
                    break

    return AuditResult("structure_binding", not viol, viol, checked, valid)


# ===========================================================================
# Orchestration
# ===========================================================================
def run_dag(narr: NarrativeOut, facts: FactContext, source: str = "llm") -> GuardrailReport:
    """Run the four narrative auditors and return the combined report with the numeric consistency rate."""
    text = narr.as_text()
    a_num = audit_numeric(text, facts)
    a_ana = audit_anatomy(text)
    a_con = audit_contradiction(narr, facts)
    a_str = audit_structure_binding(text, facts)
    audits = [a_num, a_ana, a_con, a_str]
    ncr = (a_num.valid / a_num.checked) if a_num.checked else 1.0
    return GuardrailReport(all(a.passed for a in audits), source, audits, round(ncr, 4))


# ===========================================================================
# Deterministic fallback narrative
# ===========================================================================
def _fmt(x: Any, nd: int = 2) -> str:
    """Format a number with nd decimals, or 'n/a' when missing."""
    return f"{float(x):.{nd}f}" if x is not None else "n/a"


def render_fallback(facts: FactContext) -> dict:
    """Render a template narrative from the fact context alone; it always passes `run_dag`."""
    pa, sx, cc = facts.prediction_anchor, facts.spatial_xai_anchor, facts.clinical_covariates
    p = pa.relative_probabilities
    plane = (sx.primary_plane or "coronal").lower()
    share = sx.plane_attention_share_pct.get(sx.primary_plane) if sx.primary_plane else None

    vf = (
        f"The model ranks this session as {pa.argmax_class} "
        f"(relative ranking CN {_fmt(p.get('CN'))} / MCI {_fmt(p.get('MCI'))} / AD {_fmt(p.get('AD'))}; "
        f"these are ordinal ranks, not calibrated confidence). "
        f"Saliency ({sx.xai_method}) is concentrated on the {plane} plane"
        + (f", which carries {_fmt(share, 1)} percent of inter-slice attention." if share is not None else ".")
    )
    if sx.aseg_available and sx.on_target_energy_pct is not None:
        struct = ", ".join(sx.dominant_structures or ["hippocampus", "lateral ventricle"])
        vf += (
            f" Of the attention-weighted saliency energy, {_fmt(sx.on_target_energy_pct)} percent falls within "
            f"the measured atrophy target ({struct}), about {_fmt(sx.ebpg_concentration_ratio)} times the "
            f"uniform baseline of {_fmt(sx.baseline_uniform_pct)} percent."
        )
        if sx.dominant_structures:
            vf += f" {sx.structure_saliency_caveat}"
    else:
        vf += " No subject segmentation was available, so on-target saliency energy is not reported."

    parts = []
    if cc.age is not None:
        parts.append(f"age {_fmt(cc.age, 0)}")
    if cc.gender:
        parts.append(str(cc.gender).lower())
    if cc.mmse_available and cc.mmse_score is not None:
        parts.append(f"MMSE {_fmt(cc.mmse_score, 0)}")
    if cc.cdr_score is not None:
        parts.append(f"CDR {_fmt(cc.cdr_score, 1)}")
    if cc.apoe_available and cc.apoe_genotype:
        parts.append(f"APOE genotype {cc.apoe_genotype}")
    mc = (
        "Recorded covariates (" + ", ".join(parts) + ") are shown as read-only context and were not "
        "inputs to this decision. APOE genotype is encoded as a single e4-presence flag, without "
        "accounting for the protective e2 allele, and is not treated as evidence for or against "
        "Alzheimer's disease here."
        if parts
        else "No covariates were supplied; the decision is image-only."
    )

    dj = (
        f"Argmax over the model output favours {pa.argmax_class} over the other classes. "
        f"The covariate branch is masked in the primary decision path, so this justification rests on the "
        f"imaging attribution above."
    )

    if pa.ood_status == "outside_validated_domain":
        un = (
            "This session lies OUTSIDE the validated range (squared Mahalanobis distance "
            f"{_fmt(pa.sq_mahalanobis_distance, 1)} against a threshold of {_fmt(pa.ood_threshold, 1)}). "
            "The ranking should be treated as unreliable and confirmed clinically."
        )
    elif pa.ood_status == "in_validated_domain":
        un = (
            "This session is within the validated range (squared Mahalanobis distance "
            f"{_fmt(pa.sq_mahalanobis_distance, 1)} against a threshold of {_fmt(pa.ood_threshold, 1)}), "
            "but the model is validated only for FreeSurfer-processed OASIS-3 MRI and remains decision support, "
            "not a diagnosis."
        )
    else:
        un = (
            "Out-of-domain status is unavailable for this session; the output is decision support only and the "
            "model is validated only for FreeSurfer-processed OASIS-3 MRI."
        )

    return {
        "visual_findings": vf,
        "multimodal_correlation": mc,
        "differential_justification": dj,
        "differential_conclusion": pa.argmax_class,
        "uncertainty_note": un,
    }


def flatten_narrative(narrative: dict) -> str:
    """Combine the 5 sections into a single text for `DiagnosticReport.conclusion`."""
    order = [
        ("Findings", "visual_findings"),
        ("Multimodal correlation", "multimodal_correlation"),
        ("Differential justification", "differential_justification"),
        ("Impression", "differential_conclusion"),
        ("Uncertainty", "uncertainty_note"),
    ]
    return "\n".join(f"{label}: {narrative.get(key, '').strip()}" for label, key in order if narrative.get(key))


# ===========================================================================
# Q&A mode guardrails. The narrative DAG above is unchanged; everything below is used only by
# `run_qa_dag()` and `qa_engine.py`, and is versioned separately (QA_GUARDRAIL_VERSION).
# ===========================================================================
REFUSAL_TOKEN = "<REFUSE>"

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"
_NLI_ENTAILMENT_THRESHOLD = 0.6  # minimum entailment probability for a grounded answer
# Generic NLI is weak on premises with numeric comparisons, so comparison and superlative
# answers are additionally verified arithmetically (audit_comparison, audit_superlative).


@functools.lru_cache(maxsize=1)
def _nli_model():
    """Return the local NLI cross-encoder, loaded lazily on first use and then cached."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(_NLI_MODEL_NAME)


_TRAILING_ZERO_RX = re.compile(r"(?<=\d)\.0\b")


def _normalize_for_nli(text: str) -> str:
    """Drop the redundant '.0' of whole numbers ('30.0' -> '30') before NLI scoring.

    Applied to both premise and hypothesis, because the NLI model is sensitive to this
    lexical difference.
    """
    return _TRAILING_ZERO_RX.sub("", text)


def nli_entailment_scores_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """P(entailment) for many (premise, hypothesis) pairs in a single model call.

    An empty list returns an empty list without loading the model.
    """
    import numpy as np

    if not pairs:
        return []
    model = _nli_model()
    normed = [(_normalize_for_nli(p), _normalize_for_nli(h)) for p, h in pairs]
    logits = model.predict(normed)
    label2idx = {v: k for k, v in model.config.id2label.items()}
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    idx = label2idx["entailment"]
    return [float(row[idx]) for row in probs]


def nli_entailment_score(premise: str, hypothesis: str) -> float:
    """P(entailment) for one (premise, hypothesis) pair from the local 3-class NLI model."""
    return nli_entailment_scores_batch([(premise, hypothesis)])[0]


def _fmt_evidence_value(val: Any) -> str:
    """Render a fact value as natural text for an NLI premise.

    Whole floats lose their '.0', and dictionaries become 'key value, key value'.
    """
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    if isinstance(val, dict):
        return ", ".join(f"{k} {_fmt_evidence_value(v)}" for k, v in val.items())
    return str(val)


# ===========================================================================
# Per-field faithfulness of one-shot narratives. This is a reporting metric (used by
# tools/evaluate_llm_paper.py), not part of `run_dag()`: it never changes a pass/fail verdict.
# ===========================================================================
_NARRATIVE_FAITHFULNESS_FIELDS = [
    "prediction_anchor.argmax_class",
    "prediction_anchor.relative_probabilities",
    "prediction_anchor.ood_status",
    "prediction_anchor.sq_mahalanobis_distance",
    "prediction_anchor.ood_threshold",
    "spatial_xai_anchor.primary_plane",
    "spatial_xai_anchor.plane_attention_share_pct",
    "spatial_xai_anchor.on_target_energy_pct",
    "spatial_xai_anchor.baseline_uniform_pct",
    "spatial_xai_anchor.dominant_structures",
    "clinical_covariates.mmse_score",
    "clinical_covariates.cdr_score",
    "clinical_covariates.apoe_genotype",
    "clinical_covariates.age",
    "clinical_covariates.gender",
]


# Narrative section that discusses each fact field (as fixed by the system prompt); it provides
# the NLI premise for that field.
_FIELD_TO_NARRATIVE_SECTION: dict[str, str] = {
    "prediction_anchor.argmax_class": "differential_justification",
    "prediction_anchor.relative_probabilities": "differential_justification",
    "prediction_anchor.ood_status": "uncertainty_note",
    "prediction_anchor.sq_mahalanobis_distance": "uncertainty_note",
    "prediction_anchor.ood_threshold": "uncertainty_note",
    "spatial_xai_anchor.primary_plane": "visual_findings",
    "spatial_xai_anchor.plane_attention_share_pct": "visual_findings",
    "spatial_xai_anchor.on_target_energy_pct": "visual_findings",
    "spatial_xai_anchor.baseline_uniform_pct": "visual_findings",
    "spatial_xai_anchor.dominant_structures": "visual_findings",
    "clinical_covariates.mmse_score": "multimodal_correlation",
    "clinical_covariates.cdr_score": "multimodal_correlation",
    "clinical_covariates.apoe_genotype": "multimodal_correlation",
    "clinical_covariates.age": "multimodal_correlation",
    "clinical_covariates.gender": "multimodal_correlation",
}

_OOD_STATUS_HYPOTHESIS = {
    "outside_validated_domain": "The scan is outside the model's validated domain.",
    "in_validated_domain": "The scan is within the model's validated domain.",
    "unavailable": "Out-of-domain status is unavailable for this scan.",
}


def _faithfulness_hypothesis(path: str, val: Any) -> str:
    """Phrase one fact as a natural-language sentence, used as the NLI hypothesis."""
    leaf = path.rsplit(".", 1)[-1]
    if leaf == "argmax_class":
        return f"The predicted class is {val}."
    if leaf == "relative_probabilities" and isinstance(val, dict):
        parts = ", ".join(f"{k} at {_fmt_evidence_value(v)}" for k, v in val.items())
        return f"The relative probabilities are {parts}."
    if leaf == "ood_status":
        return _OOD_STATUS_HYPOTHESIS.get(val, f"The out-of-domain status is {val}.")
    if leaf == "sq_mahalanobis_distance":
        return f"The squared Mahalanobis distance is {_fmt_evidence_value(val)}."
    if leaf == "ood_threshold":
        return f"The out-of-domain threshold is {_fmt_evidence_value(val)}."
    if leaf == "primary_plane":
        return f"The primary attention plane is {val}."
    if leaf == "plane_attention_share_pct" and isinstance(val, dict):
        parts = ", ".join(f"{k} at {_fmt_evidence_value(v)}%" for k, v in val.items())
        return f"Plane attention shares are {parts}."
    if leaf == "on_target_energy_pct":
        return f"The on-target saliency energy is {_fmt_evidence_value(val)}%."
    if leaf == "baseline_uniform_pct":
        return f"The baseline uniform percentage is {_fmt_evidence_value(val)}%."
    if leaf == "dominant_structures":
        return f"The dominant structures are {', '.join(val) if val else 'none recorded'}."
    if leaf == "mmse_score":
        return f"The MMSE score is {_fmt_evidence_value(val)}."
    if leaf == "cdr_score":
        return f"The CDR score is {_fmt_evidence_value(val)}."
    if leaf == "apoe_genotype":
        return f"The APOE genotype is {val}."
    if leaf == "age":
        return f"The patient is {_fmt_evidence_value(val)} years old."
    if leaf == "gender":
        return f"The patient is {val}."
    return f"{leaf.replace('_', ' ')} is {_fmt_evidence_value(val)}."  # generic fallback


def narrative_faithfulness_per_field(narr: NarrativeOut, facts: FactContext) -> dict[str, float]:
    """Per-field NLI faithfulness of a narrative to the fact context.

    For each fact field the premise is the narrative section that should discuss it
    (`_FIELD_TO_NARRATIVE_SECTION`) and the hypothesis is a natural sentence stating the fact,
    so the score is P(section entails fact). Fields without a value, or whose section is
    empty, are omitted. All pairs are scored in one batched call.

    A low score for fields the prompt does not ask the model to verbalize (squared Mahalanobis
    distance, out-of-domain threshold) means "not verified from the text", not an error in the
    narrative.
    """
    paths: list[str] = []
    pairs: list[tuple[str, str]] = []
    for path in _NARRATIVE_FAITHFULNESS_FIELDS:
        val = resolve_fact_path(facts, path)
        if val is None:
            continue
        section = _FIELD_TO_NARRATIVE_SECTION.get(path)
        premise = getattr(narr, section, None) if section else None
        if not premise:
            continue
        paths.append(path)
        pairs.append((premise, _faithfulness_hypothesis(path, val)))
    scores = nli_entailment_scores_batch(pairs)
    return dict(zip(paths, scores))


def narrative_faithfulness_score(narr: NarrativeOut, facts: FactContext) -> float:
    """Mean of `narrative_faithfulness_per_field()`; 0.0 when no field could be scored."""
    scores = narrative_faithfulness_per_field(narr, facts)
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def audit_grounding(a: "AnswerOut", fc: FactContext, source: str = "llm",
                     threshold: float = _NLI_ENTAILMENT_THRESHOLD) -> AuditResult:
    """Check that an answer is grounded in the fact fields it cites.

    (a) Every `cited_fields` entry must resolve to a fact path.
    (b) NLI entailment of the answer by the combined cited values must reach the threshold.

    Deterministic refusals and the refusal token pass trivially: refusals are generated from the
    same facts, and an empty citation list is valid for them.
    """
    if source == "deterministic_refusal" or a.answer.strip() == REFUSAL_TOKEN:
        return AuditResult("grounding", True, [], checked=0, valid=0)

    viol: list[Violation] = []
    evidence_parts: list[str] = []
    for cf in a.cited_fields:
        val = resolve_fact_path(fc, cf)
        if val is None:
            viol.append(Violation("unresolvable_citation",
                                  f"cited_field '{cf}' does not resolve to a Fact Context path", found=cf))
        else:
            evidence_parts.append(f"{cf}: {_fmt_evidence_value(val)}")

    if not a.cited_fields:
        viol.append(Violation("missing_citation", "cited_fields is empty for a non-refusal answer"))
    elif not viol:
        premise = ". ".join(evidence_parts)
        score = nli_entailment_score(premise, a.answer)
        if score < threshold:
            viol.append(Violation(
                "ungrounded_answer",
                f"answer not entailed by its own cited_fields (entailment={score:.2f} < {threshold})",
                found=f"{score:.2f}",
            ))
    return AuditResult("grounding", not viol, viol, checked=1, valid=0 if viol else 1)


def audit_scope(a: "AnswerOut", category: str, source: str = "llm") -> AuditResult:
    """Check that out-of-scope and unanswerable questions end in a refusal.

    A deterministic refusal only needs `answerable=False`; a raw LLM draft must be exactly the
    refusal token. Other categories pass trivially.
    """
    if category not in ("out_of_scope", "unanswerable"):
        return AuditResult("scope", True, [], checked=0, valid=0)
    if source == "deterministic_refusal":
        ok = a.answerable is False
    else:
        ok = a.answerable is False and a.answer.strip() == REFUSAL_TOKEN
    viol = [] if ok else [Violation(
        "scope_violation", f"category='{category}' requires a deterministic refusal", found=a.answer[:80])]
    return AuditResult("scope", not viol, viol, checked=1, valid=0 if viol else 1)


def _value_matches(num: float, true_val: float) -> bool:
    """Compare a number written in text with a fact value on either scale (fraction or percent)."""
    if abs(true_val) <= 1.5:  # most likely a fraction (e.g. a 0-1 probability)
        candidates = [(true_val, 0.02), (true_val * 100, 0.7)]
    else:  # most likely already percent scale
        candidates = [(true_val, 0.7), (true_val / 100, 0.02)]
    return any(abs(num - cand) <= tol for cand, tol in candidates)


_LABEL_NUM_CONN_RX = (
    r"[\s:=\(]{0,2}(?:is\s+at|carries|reaches|shows|has|was|is|at|of|for|to|in)?[\s:=\(]{0,2}"
)


def _extract_label_value_pairs(text: str, labels: list[str]) -> list[tuple[str, float]]:
    """Find (label, number) pairs in which the number sits directly next to the label.

    Only a short connector (':', '=', '(', 'is', 'at', 'of', ...) may separate them; labels
    without an adjacent number are skipped to favour precision.
    """
    pairs: list[tuple[str, float]] = []
    for label in labels:
        esc = re.escape(label)
        for m in re.finditer(rf"\b{esc}\b{_LABEL_NUM_CONN_RX}(\d+(?:\.\d+)?)\s*%?", text, re.I):
            pairs.append((label, float(m.group(1))))
        for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*%?{_LABEL_NUM_CONN_RX}\b{esc}\b", text, re.I):
            pairs.append((label, float(m.group(1))))
    return pairs


def audit_label_binding(a: "AnswerOut", fc: FactContext) -> AuditResult:
    """Check that numbers written next to a label match that label's value (detects swapped pairs).

    Covers cited fields whose value is a flat label-to-number dictionary (e.g.
    `relative_probabilities`, `plane_attention_share_pct`); other fields are skipped.
    """
    if a.answer.strip() == REFUSAL_TOKEN:
        return AuditResult("label_binding", True, [], checked=0, valid=0)

    viol: list[Violation] = []
    checked = valid = 0
    for cf in a.cited_fields:
        val = resolve_fact_path(fc, cf)
        if not isinstance(val, dict) or not val or not all(isinstance(v, (int, float)) for v in val.values()):
            continue
        for label, num in _extract_label_value_pairs(a.answer, list(val.keys())):
            checked += 1
            true_val = float(val[label])
            if _value_matches(num, true_val):
                valid += 1
                continue
            swapped_with = next(
                (other for other, ov in val.items() if other != label and _value_matches(num, float(ov))),
                None,
            )
            detail = f"'{label}' paired with {num}, but its true value in '{cf}' is {true_val}"
            if swapped_with:
                detail += f" (this number belongs to '{swapped_with}' -- looks swapped)"
            viol.append(Violation("label_value_mismatch", detail, found=f"{label}={num}", expected=f"{label}={true_val}"))
    return AuditResult("label_binding", not viol, viol, checked, valid)


_SUPERLATIVE_MAX_RX = re.compile(r"\b(?:most|highest|greatest|largest|top(?:-|\s)?ranked|maximum)\b", re.I)
_SUPERLATIVE_MIN_RX = re.compile(r"\b(?:least|lowest|smallest|minimum)\b", re.I)


def audit_superlative(a: "AnswerOut", fc: FactContext) -> AuditResult:
    """Check "highest/most" and "lowest/least" claims about a labelled field against its extremes.

    For each sentence containing a superlative word and exactly one label of a cited dictionary
    field, the label must be that field's maximum (or minimum). Sentences with no label or with
    several labels are skipped as unverifiable. No model is involved.
    """
    if a.answer.strip() == REFUSAL_TOKEN:
        return AuditResult("superlative", True, [], checked=0, valid=0)

    sentences = [s for s in a.answer.split(".") if _SUPERLATIVE_MAX_RX.search(s) or _SUPERLATIVE_MIN_RX.search(s)]
    if not sentences:
        return AuditResult("superlative", True, [], checked=0, valid=0)

    viol: list[Violation] = []
    checked = valid = 0
    for cf in a.cited_fields:
        val = resolve_fact_path(fc, cf)
        if not isinstance(val, dict) or not val or not all(isinstance(v, (int, float)) for v in val.values()):
            continue
        for sent in sentences:
            present = [lbl for lbl in val if re.search(rf"\b{re.escape(lbl)}\b", sent, re.I)]
            if len(present) != 1:
                continue  # no label, or several (ambiguous)
            label = present[0]
            # A sentence may contain both a maximum and a minimum word; verify each independently
            for rx, kind, pick in (
                (_SUPERLATIVE_MAX_RX, "highest", max),
                (_SUPERLATIVE_MIN_RX, "lowest", min),
            ):
                if not rx.search(sent):
                    continue
                checked += 1
                target = pick(val, key=val.get)
                if label == target:
                    valid += 1
                    continue
                viol.append(Violation(
                    "superlative_mismatch",
                    f"answer claims '{label}' is {kind} in '{cf}', but '{target}' actually is",
                    found=label, expected=target,
                ))
    return AuditResult("superlative", not viol, viol, checked, valid)


def _apply_superlative_override(a_grd: AuditResult, a_sup: AuditResult) -> AuditResult:
    """Let a correct, active superlative check override a failed NLI grounding verdict.

    A silent superlative check (`checked == 0`) leaves the grounding result unchanged.
    """
    if a_grd.passed or not (a_sup.checked > 0 and a_sup.passed):
        return a_grd
    return AuditResult(a_grd.name, True, [], a_grd.checked, a_grd.valid)


# "number <comparison word> number" claims. Gaps are bounded runs of non-digit characters so
# that a different number is never skipped over.
_GT_CLAIM_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*%?(?:(?!\d)[^.]){0,30}?"
    r"(?:greater than|more than|higher than|larger than|exceed(?:s|ing)?|above)\b"
    r"(?:(?!\d)[^.]){0,40}?(\d+(?:\.\d+)?)\s*%?",
    re.I,
)
_LT_CLAIM_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*%?(?:(?!\d)[^.]){0,30}?"
    r"(?:less than|lower than|smaller than|below|under)\b"
    r"(?:(?!\d)[^.]){0,40}?(\d+(?:\.\d+)?)\s*%?",
    re.I,
)


def audit_comparison(a: "AnswerOut") -> AuditResult:
    """Check arithmetic comparison claims ("X is greater/smaller than Y") written in the answer.

    Verifies only the ordering between the two numbers the answer itself states, without
    consulting the fact context. Claims without a literal number on both sides are not checked,
    and negation is not handled.
    """
    if a.answer.strip() == REFUSAL_TOKEN:
        return AuditResult("comparison", True, [], checked=0, valid=0)

    checked = valid = 0
    viol: list[Violation] = []
    for rx, claim_op, symbol in ((_GT_CLAIM_RX, "gt", ">"), (_LT_CLAIM_RX, "lt", "<")):
        for m in rx.finditer(a.answer):
            left, right = float(m.group(1)), float(m.group(2))
            checked += 1
            ok = (left > right) if claim_op == "gt" else (left < right)
            if ok:
                valid += 1
            else:
                viol.append(Violation(
                    "comparison_direction_mismatch",
                    f"answer claims {left} {symbol} {right}, but that is arithmetically false",
                    found=m.group(0).strip(),
                ))
    return AuditResult("comparison", not viol, viol, checked, valid)


def _apply_comparison_override(a_grd: AuditResult, a_cmp: AuditResult) -> AuditResult:
    """Let a correct, active comparison check override a failed NLI grounding verdict.

    Generic NLI often rejects correct arithmetic comparisons. A silent comparison check
    (`checked == 0`) or an already passing grounding result is returned unchanged.
    """
    if a_grd.passed or not (a_cmp.checked > 0 and a_cmp.passed):
        return a_grd
    return AuditResult(a_grd.name, True, [], a_grd.checked, a_grd.valid)


def run_qa_dag(a: "AnswerOut", fc: FactContext, base: NarrativeOut, category: str,
               source: str = "llm") -> GuardrailReport:
    """Run the Q&A auditors on an answer and return the combined report.

    Reuses the narrative auditors (numeric, anatomy, structure binding, contradiction) on the
    answer text; contradiction runs on a shim narrative pinned to the base conclusion, since a
    Q&A answer never states a new diagnosis. Adds grounding, scope, label binding, comparison
    and superlative auditors. A failed grounding verdict may be overridden by a correct
    comparison or superlative check. The report carries QA_GUARDRAIL_VERSION.
    """
    text = a.answer
    a_num = audit_numeric(text, fc)
    a_ana = audit_anatomy(text)
    a_str = audit_structure_binding(text, fc)
    shim = NarrativeOut(
        visual_findings=text, multimodal_correlation="", differential_justification="",
        differential_conclusion=base.differential_conclusion, uncertainty_note="",
    )
    a_con = audit_contradiction(shim, fc)
    a_grd = audit_grounding(a, fc, source)
    a_scp = audit_scope(a, category, source)
    a_lbl = audit_label_binding(a, fc)
    a_cmp = audit_comparison(a)
    a_sup = audit_superlative(a, fc)
    a_grd = _apply_comparison_override(a_grd, a_cmp)
    a_grd = _apply_superlative_override(a_grd, a_sup)
    audits = [a_num, a_ana, a_str, a_con, a_grd, a_scp, a_lbl, a_cmp, a_sup]
    ncr = (a_num.valid / a_num.checked) if a_num.checked else 1.0
    return GuardrailReport(all(x.passed for x in audits), source, audits, round(ncr, 4), version=QA_GUARDRAIL_VERSION)


_REFUSAL_MESSAGES = {
    "out_of_scope": (
        "This question asks for a clinical recommendation (e.g. treatment, further "
        "testing, or prognosis), which is outside what this decision-support narrative "
        "reports. It only explains the existing image-based prediction."
    ),
    "unanswerable": (
        "This question needs information that was not measured or recorded for this "
        "session, so it cannot be answered from the available evidence."
    ),
}

_FIELD_LABELS = {
    "mmse_score": "MMSE score", "cdr_score": "CDR score", "apoe_genotype": "APOE genotype",
    "argmax_class": "predicted class",
    # The acronym is used on purpose: "out-of-domain" in a label would trigger `_OOD_OUT_RX`
    "ood_status": "OOD status",
    "sq_mahalanobis_distance": "OOD distance", "primary_plane": "dominant imaging plane",
    "on_target_energy_pct": "on-target saliency energy",
}


def _humanize_field(path: str) -> str:
    """Readable label for a fact path ('clinical_covariates.mmse_score' -> 'MMSE score')."""
    leaf = path.rsplit(".", 1)[-1]
    return _FIELD_LABELS.get(leaf, leaf.replace("_", " "))


def render_refusal(category: str, target_fields: list[str], fc: FactContext) -> "AnswerOut":
    """Build a deterministic refusal that restates up to two recorded facts; it passes `run_qa_dag`."""
    from .qa_engine import AnswerOut  # local import avoids a circular import

    restated: list[str] = []
    cited: list[str] = []
    for tf in target_fields[:2]:
        val = resolve_fact_path(fc, tf)
        if val is not None:
            restated.append(f"{_humanize_field(tf)} is {val}")
            cited.append(tf)
    message = _REFUSAL_MESSAGES.get(category, "This question cannot be answered from the recorded session facts.")
    if restated:
        message += " What is on file: " + "; ".join(restated) + "."
    return AnswerOut(answer=message, cited_fields=cited, answerable=False)


class CoherenceError(RuntimeError):
    """Raised when a Q&A turn must not be kept as is (class shift or turn cap)."""


MAX_QA_TURNS = 12  # maximum turns per Q&A session

_OTHER_CLASS_CLAIM_RX = {
    "AD": re.compile(r"\balzheimer'?s?\b|\bAD\b", re.I),
    "MCI": re.compile(r"\bmild cognitive impairment\b|\bMCI\b", re.I),
    "CN": re.compile(r"\bcognitively normal\b|\bCN\b", re.I),
}
_CLASS_ASSERT_RX = re.compile(
    r"\b(is|indicates?|consistent with|diagnos\w+ (?:of|as)|really|actually|basically|confirms?)\b", re.I,
)
_NEGATION_RX = re.compile(r"\b(not|n't|rather than|instead of|ruled out|no evidence of|against)\b", re.I)


def coherence_guard(turn: "QATurn", qa: "QASession") -> None:  # noqa: F821
    """Raise CoherenceError when an answer appears to assert a class other than the pinned conclusion.

    A regex heuristic that skips matches with a nearby negation marker. Each turn is checked
    against the pinned base narrative only, never against earlier turns.
    """
    if turn.answer_out is None or turn.answer_out.answer.strip() == REFUSAL_TOKEN:
        return
    pinned = qa.base_narrative.differential_conclusion
    text = turn.answer_out.answer
    for cls, rx in _OTHER_CLASS_CLAIM_RX.items():
        if cls == pinned:
            continue
        m = rx.search(text)
        if not m:
            continue
        window = text[max(0, m.start() - 40): m.start()]
        if _NEGATION_RX.search(window):
            continue  # negated mention ("not AD", "AD ruled out")
        if _CLASS_ASSERT_RX.search(text):
            raise CoherenceError(
                f"answer appears to assert class '{cls}' while the pinned differential_conclusion is '{pinned}'"
            )
