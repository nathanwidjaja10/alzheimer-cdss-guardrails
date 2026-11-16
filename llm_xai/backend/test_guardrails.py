"""Regression tests for the narrative and Q&A guardrails.

Narrative fixture: a draft that contains a fabricated structure ("inferior parietal lobule")
and uses APOE4 as evidence for AD across two sentences of one field; the DAG must reject it.
Further tests cover APOE negation, slice and structure binding, arithmetic comparison claims,
superlative claims and the NLI override logic.

Run: `python -m pytest llm_xai/backend/test_guardrails.py -q`
(or `python llm_xai/backend/test_guardrails.py`, which does not need pytest).
"""
from __future__ import annotations

from .guardrails import (
    AuditResult,
    Violation,
    _apply_comparison_override,
    _apply_superlative_override,
    audit_anatomy,
    audit_comparison,
    audit_contradiction,
    audit_structure_binding,
    audit_superlative,
    run_dag,
)
from .llm_engine import (
    ClinicalCovariates,
    FactContext,
    ModelProvenance,
    NarrativeOut,
    PredictionAnchor,
    SpatialXaiAnchor,
    TopSlice,
)

FACTS = FactContext(
    session_id="OAS30002_MR_d2340",
    prediction_anchor=PredictionAnchor(
        argmax_class="MCI",
        relative_probabilities={"CN": 0.1, "MCI": 0.7, "AD": 0.2},
        ood_status="outside_validated_domain",
        sq_mahalanobis_distance=99.0,
        ood_threshold=12.0,
    ),
    spatial_xai_anchor=SpatialXaiAnchor(primary_plane="coronal", aseg_available=False),
    clinical_covariates=ClinicalCovariates(
        age=73, gender="male", mmse_score=30, apoe_genotype="e3/e4", apoe_available=True,
    ),
    model_provenance=ModelProvenance(),
)

# Narrative with a fabricated structure and an APOE4-as-evidence statement
NARR_BAD = NarrativeOut(
    visual_findings=(
        "The saliency is in the medial temporal lobe, including the hippocampus and "
        "lateral ventricles. There is also a small focus of atrophy in the inferior "
        "parietal lobule."
    ),
    multimodal_correlation=(
        "This patient has mild cognitive impairment (MCI) with an MMSE score of "
        "30/30, but no CDR score recorded. The APOE4 allele is present. This is "
        "consistent with a diagnosis of Alzheimer's disease (AD)."
    ),
    differential_justification=(
        "The patient has mild cognitive impairment and an APOE4 allele, which are "
        "risk factors for AD. The saliency is in the medial temporal lobe, including "
        "the hippocampus and lateral ventricles, which is consistent with AD "
        "pathology. There is also a small focus of atrophy in the inferior parietal "
        "lobule, which is not a typical finding in AD. Overall, this patient has an "
        "MCI phenotype that is consistent with AD, rather than other causes of MCI."
    ),
    differential_conclusion="MCI",
    uncertainty_note=(
        "The model is validated only for OASIS-3 structural MRI processed with "
        "FreeSurfer. Scans from other sites or pipelines fall outside the validated "
        "range and are flagged by the out-of-domain check."
    ),
)


def test_fabricated_structure_caught() -> None:
    result = audit_anatomy(NARR_BAD.as_text())
    types = {v.type for v in result.violations}
    assert "fabricated_structure" in types, (
        "'inferior parietal lobule' must be caught as fabricated_structure (tier C)"
    )
    assert any(v.found == "inferior parietal lobule" for v in result.violations)


def test_apoe_self_diagnosis_caught() -> None:
    result = audit_contradiction(NARR_BAD, FACTS)
    types = {v.type for v in result.violations}
    assert "covariate_caveat_breach" in types, (
        "APOE4 used as evidence for AD across sentences must be caught"
    )


def test_full_dag_rejects_known_bad_case() -> None:
    report = run_dag(NARR_BAD, FACTS, source="llm_regenerated")
    assert not report.passed, (
        "run_dag must reject a narrative with a fabricated structure and an APOE self-diagnosis"
    )


# ===========================================================================
# APOE negation, slice binding and dominant-structure binding
# ===========================================================================

def test_apoe_negation_contradiction_caught() -> None:
    """A narrative that denies the E4 allele although the genotype ('e3/e4') contains it is flagged."""
    result = audit_contradiction(
        NarrativeOut(
            visual_findings="Saliency is concentrated in the coronal plane.",
            multimodal_correlation="There is no APOE4 allele available for consideration.",
            differential_justification="The relative ranking favours MCI over CN and AD.",
            differential_conclusion="MCI",
            uncertainty_note="This scan falls outside the validated range.",
        ),
        FACTS,
    )
    types = {v.type for v in result.violations}
    assert "apoe_negation_contradiction" in types, (
        "'no APOE4 allele' must be caught when the genotype contains E4 ('e3/e4')"
    )


def test_apoe_negation_safe_phrasing_not_flagged() -> None:
    """"No indication that the allele provides evidence" is epistemic phrasing, not a denial of the allele."""
    result = audit_contradiction(
        NarrativeOut(
            visual_findings="Saliency is concentrated in the coronal plane.",
            multimodal_correlation=(
                "There is no indication that the APOE4 allele provides evidence "
                "for or against any class."
            ),
            differential_justification="The relative ranking favours MCI over CN and AD.",
            differential_conclusion="MCI",
            uncertainty_note="This scan falls outside the validated range.",
        ),
        FACTS,
    )
    types = {v.type for v in result.violations}
    assert "apoe_negation_contradiction" not in types, (
        "'no indication that X provides evidence' must not be treated as 'no APOE4 allele'"
    )


FACTS_SLICES = FACTS.model_copy(deep=True)
FACTS_SLICES.spatial_xai_anchor.top_slices = [
    TopSlice(name="Coronal", index=4, attention_share_pct=61.2),
    TopSlice(name="Coronal", index=5, attention_share_pct=24.8),
]
FACTS_SLICES.spatial_xai_anchor.dominant_structures = ["hippocampus"]


def test_slice_binding_swapped_caught() -> None:
    """Two correctly named slices with swapped percentages pass `audit_numeric` but are caught by `audit_structure_binding`."""
    text = "Coronal 4 (24.8%) and Coronal 5 (61.2%) show the highest attention share."
    result = audit_structure_binding(text, FACTS_SLICES)
    types = {v.type for v in result.violations}
    assert "slice_binding_mismatch" in types


def test_slice_binding_correct_passes() -> None:
    text = "Coronal 4 (61.2%) and Coronal 5 (24.8%) show the highest attention share."
    result = audit_structure_binding(text, FACTS_SLICES)
    assert result.passed and result.checked == 2 and result.valid == 2


def test_dominant_structure_mismatch_caught() -> None:
    """Naming a whitelisted structure that is not in `dominant_structures` as the focus of attention is flagged."""
    text = "The model's attention being focused on cerebral cortex and other structures."
    result = audit_structure_binding(text, FACTS_SLICES)
    types = {v.type for v in result.violations}
    assert "dominant_structure_mismatch" in types


def test_dominant_structure_correct_passes() -> None:
    text = "The model's attention was most concentrated in the hippocampus."
    result = audit_structure_binding(text, FACTS_SLICES)
    assert result.passed
    assert not any(v.type == "dominant_structure_mismatch" for v in result.violations)


# ===========================================================================
# Arithmetic comparison claims (audit_comparison). `AnswerOut` is imported from qa_engine here
# because guardrails.py cannot import it at module level.
# ===========================================================================
from .qa_engine import AnswerOut  # noqa: E402


def test_comparison_backward_claim_caught() -> None:
    # Reversed comparison: 0.4903 > 0.4157, not "<". Pure arithmetic must catch it.
    a = AnswerOut(
        answer="MCI is ranked below CN, since 0.4903 is less than 0.4157.",
        cited_fields=["prediction_anchor.relative_probabilities"], answerable=True,
    )
    result = audit_comparison(a)
    assert not result.passed
    assert any(v.type == "comparison_direction_mismatch" for v in result.violations)


def test_comparison_threshold_claim_caught() -> None:
    # Reversed comparison with filler words between the two numbers
    a = AnswerOut(
        answer=("The distance of 1418.6 is below the out-of-domain threshold of 997.5, "
                "so this scan is within the validated domain."),
        cited_fields=["prediction_anchor.sq_mahalanobis_distance", "prediction_anchor.ood_threshold"],
        answerable=True,
    )
    result = audit_comparison(a)
    assert not result.passed
    assert any(v.type == "comparison_direction_mismatch" for v in result.violations)


def test_comparison_correct_claim_passes() -> None:
    # A correct claim must pass as verified (checked=1, valid=1), not merely as unchecked
    a = AnswerOut(
        answer="MCI is more likely than CN because 0.7859 is greater than 0.1869.",
        cited_fields=["prediction_anchor.relative_probabilities"], answerable=True,
    )
    result = audit_comparison(a)
    assert result.passed
    assert result.checked == 1 and result.valid == 1


def test_comparison_label_only_claim_skipped() -> None:
    # A claim naming labels without a number in the same sentence is not checked (checked=0)
    a = AnswerOut(
        answer="MCI is more likely than CN overall.",
        cited_fields=["prediction_anchor.relative_probabilities"], answerable=True,
    )
    result = audit_comparison(a)
    assert result.passed and result.checked == 0


def test_comparison_catches_parenthesised_number_before_keyword() -> None:
    # Number in parentheses followed by an adverb before the comparison word
    a = AnswerOut(
        answer=("The model picked AD over MCI and CN because its relative probability "
                "(0.9449) is significantly higher than those of MCI (0.0253) and CN (0.0298)."),
        cited_fields=["prediction_anchor.relative_probabilities"], answerable=True,
    )
    result = audit_comparison(a)
    assert result.passed and result.checked >= 1 and result.valid == result.checked


def test_comparison_catches_comma_and_gerund_before_keyword() -> None:
    # Comma before the comparison word and the gerund form "exceeding"
    a = AnswerOut(
        answer=("The scan is flagged as outside the validated domain because its "
                "Mahalanobis distance (1401.1) exceeds the out-of-domain threshold (997.5)."),
        cited_fields=["prediction_anchor.sq_mahalanobis_distance", "prediction_anchor.ood_threshold"],
        answerable=True,
    )
    result = audit_comparison(a)
    assert result.passed and result.checked >= 1 and result.valid == result.checked

    a2 = AnswerOut(
        answer=("The scan is outside the validated domain, with a Mahalanobis distance "
                "of 1401.1 exceeding the threshold of 997.5."),
        cited_fields=["prediction_anchor.sq_mahalanobis_distance", "prediction_anchor.ood_threshold"],
        answerable=True,
    )
    result2 = audit_comparison(a2)
    assert result2.passed and result2.checked >= 1 and result2.valid == result2.checked


# ===========================================================================
# Comparison override of a failed NLI grounding verdict. Tested at function level with
# hand-built AuditResults, so no NLI model is loaded.
# ===========================================================================
def test_comparison_override_lifts_nli_failure_when_audit_active_and_correct() -> None:
    a_grd = AuditResult("grounding", False, [Violation("ungrounded_answer", "low NLI score")], checked=1, valid=0)
    a_cmp = AuditResult("comparison", True, [], checked=1, valid=1)
    overridden = _apply_comparison_override(a_grd, a_cmp)
    assert overridden.passed, (
        "grounding must pass when audit_comparison is active and correct"
    )


def test_comparison_override_keeps_nli_failure_when_audit_silent() -> None:
    a_grd = AuditResult("grounding", False, [Violation("ungrounded_answer", "low NLI score")], checked=1, valid=0)
    a_cmp = AuditResult("comparison", True, [], checked=0, valid=0)  # silent: outside its scope
    overridden = _apply_comparison_override(a_grd, a_cmp)
    assert not overridden.passed, (
        "when audit_comparison is silent (checked=0), the NLI verdict must stand"
    )


def test_comparison_override_never_overrides_a_failed_comparison() -> None:
    a_grd = AuditResult("grounding", False, [Violation("ungrounded_answer", "low NLI score")], checked=1, valid=0)
    a_cmp = AuditResult("comparison", False, [Violation("comparison_direction_mismatch", "x")], checked=1, valid=0)
    overridden = _apply_comparison_override(a_grd, a_cmp)
    assert not overridden.passed, (
        "a failed comparison audit must never be overridden"
    )


def test_comparison_override_is_noop_when_grounding_already_passed() -> None:
    a_grd = AuditResult("grounding", True, [], checked=1, valid=1)
    a_cmp = AuditResult("comparison", True, [], checked=1, valid=1)
    overridden = _apply_comparison_override(a_grd, a_cmp)
    assert overridden is a_grd, "an already passing grounding result is returned unchanged"


# ===========================================================================
# Superlative claims (audit_superlative), using plane attention shares in a local fixture
# so the default FACTS used above stays unchanged.
# ===========================================================================
FACTS_PLANES = FACTS.model_copy(deep=True)
FACTS_PLANES.spatial_xai_anchor.plane_attention_share_pct = {
    "Coronal": 91.3, "Axial": 6.1, "Sagittal": 2.6,
}


def test_superlative_correct_argmax_claim_passes() -> None:
    a = AnswerOut(
        answer="The Coronal plane got the most attention from the model, with a share of 91.3%.",
        cited_fields=["spatial_xai_anchor.plane_attention_share_pct"], answerable=True,
    )
    result = audit_superlative(a, FACTS_PLANES)
    assert result.passed and result.checked == 1 and result.valid == 1


def test_superlative_wrong_argmax_claim_caught() -> None:
    # Adversarial: naming a plane that does not have the maximum share must be caught
    a = AnswerOut(
        answer="The Axial plane got the most attention from the model, with a share of 6.1%.",
        cited_fields=["spatial_xai_anchor.plane_attention_share_pct"], answerable=True,
    )
    result = audit_superlative(a, FACTS_PLANES)
    assert not result.passed
    assert any(v.type == "superlative_mismatch" for v in result.violations)


def test_superlative_claim_without_adjacent_number_skipped() -> None:
    # A superlative without a plane label is not checked (checked=0)
    a = AnswerOut(
        answer="The model's attention was most concentrated in one particular plane overall.",
        cited_fields=["spatial_xai_anchor.plane_attention_share_pct"], answerable=True,
    )
    result = audit_superlative(a, FACTS_PLANES)
    assert result.passed and result.checked == 0


def test_superlative_override_lifts_nli_failure_when_correct() -> None:
    a_grd = AuditResult("grounding", False, [Violation("ungrounded_answer", "low NLI score")], checked=1, valid=0)
    a_sup = AuditResult("superlative", True, [], checked=1, valid=1)
    overridden = _apply_superlative_override(a_grd, a_sup)
    assert overridden.passed


def test_superlative_override_never_overrides_a_wrong_claim() -> None:
    a_grd = AuditResult("grounding", False, [Violation("ungrounded_answer", "low NLI score")], checked=1, valid=0)
    a_sup = AuditResult("superlative", False, [Violation("superlative_mismatch", "x")], checked=1, valid=0)
    overridden = _apply_superlative_override(a_grd, a_sup)
    assert not overridden.passed, (
        "a failed superlative audit must never be overridden"
    )


if __name__ == "__main__":
    test_fabricated_structure_caught()
    test_apoe_self_diagnosis_caught()
    test_full_dag_rejects_known_bad_case()
    test_apoe_negation_contradiction_caught()
    test_apoe_negation_safe_phrasing_not_flagged()
    test_slice_binding_swapped_caught()
    test_slice_binding_correct_passes()
    test_dominant_structure_mismatch_caught()
    test_dominant_structure_correct_passes()
    test_comparison_backward_claim_caught()
    test_comparison_threshold_claim_caught()
    test_comparison_correct_claim_passes()
    test_comparison_label_only_claim_skipped()
    test_comparison_catches_parenthesised_number_before_keyword()
    test_comparison_catches_comma_and_gerund_before_keyword()
    test_superlative_correct_argmax_claim_passes()
    test_superlative_wrong_argmax_claim_caught()
    test_superlative_claim_without_adjacent_number_skipped()
    test_superlative_override_lifts_nli_failure_when_correct()
    test_superlative_override_never_overrides_a_wrong_claim()
    test_comparison_override_lifts_nli_failure_when_audit_active_and_correct()
    test_comparison_override_keeps_nli_failure_when_audit_silent()
    test_comparison_override_never_overrides_a_failed_comparison()
    test_comparison_override_is_noop_when_grounding_already_passed()
    from . import GUARDRAIL_VERSION, QA_GUARDRAIL_VERSION
    print(f"OK -- all guardrail {GUARDRAIL_VERSION} + QA {QA_GUARDRAIL_VERSION} regressions passed.")
