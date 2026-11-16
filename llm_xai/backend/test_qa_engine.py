"""Tests for the deterministic parts of `qa_engine.py`; no LLM call is made.

Covered: the classification gate (`classify_question`: keyword rules and local embeddings),
retrieval (`retrieve_fields`), answer parsing including recovery from malformed JSON
(`parse_answer`), the Ollama answer schema, and deterministic refusals, which must always pass
`run_qa_dag`. The individual auditors are tested in `test_guardrails.py`.

`classify_question` loads a local sentence-embedding model once per process; there are no
network or Ollama calls.

Run: `python -m pytest llm_xai/backend/test_qa_engine.py -q`
(or `python llm_xai/backend/test_qa_engine.py`, which does not need pytest).
"""
from __future__ import annotations

from .guardrails import REFUSAL_TOKEN, render_refusal, run_qa_dag
from .llm_engine import (
    ClinicalCovariates,
    FactContext,
    LLMOutputError,
    ModelProvenance,
    NarrativeOut,
    PredictionAnchor,
    SpatialXaiAnchor,
    TopSlice,
)
from .qa_engine import (
    AnswerOut,
    answer_schema_for_ollama,
    audit_failure_reasons,
    classify_question,
    classify_question_debug,
    parse_answer,
    qa_policy_sha,
    retrieve_fields,
)

# CN, in-domain session with MMSE and APOE available (test_guardrails.py uses an MCI, out-of-domain session)
FACTS = FactContext(
    session_id="OAS30003_MR_d0100",
    prediction_anchor=PredictionAnchor(
        argmax_class="CN",
        relative_probabilities={"CN": 0.82, "MCI": 0.13, "AD": 0.05},
        ood_status="in_validated_domain",
        sq_mahalanobis_distance=4.2,
        ood_threshold=12.0,
    ),
    spatial_xai_anchor=SpatialXaiAnchor(primary_plane="axial", aseg_available=True),
    clinical_covariates=ClinicalCovariates(
        age=68, gender="female", mmse_score=29, apoe_genotype="e3/e3", apoe_available=True,
    ),
    model_provenance=ModelProvenance(),
)

BASE_NARRATIVE = NarrativeOut(
    visual_findings="Saliency concentrates in the hippocampus and lateral ventricles.",
    multimodal_correlation="MMSE of 29/30 is consistent with the predicted class.",
    differential_justification="Vision-only argmax ranks CN highest with no anatomical red flags.",
    differential_conclusion="CN",
    uncertainty_note="Validated only for OASIS-3 structural MRI processed with FreeSurfer.",
)


# ===========================================================================
# [1] classify_question / classify_question_debug
# ===========================================================================
def test_classify_out_of_scope_keyword_gate() -> None:
    # A keyword rule ("recommend") decides this without the embedding model
    category, target_fields, gate = classify_question(
        "What treatment do you recommend for this patient?", FACTS
    )
    assert category == "out_of_scope"
    assert target_fields == []
    assert gate is False


def test_classify_retrieval_question_via_embedding() -> None:
    # Paraphrase of a bank example, so the embedding path is exercised
    category, target_fields, gate = classify_question(
        "What's this patient's MMSE score on file?", FACTS
    )
    assert category == "retrieval"
    assert "clinical_covariates.mmse_score" in target_fields
    assert gate is True


def test_classify_leading_question_via_embedding() -> None:
    category, _target_fields, gate = classify_question(
        "C'mon, this is obviously Alzheimer's, right?", FACTS
    )
    assert category == "leading"
    assert gate is True


def test_classify_unrelated_question_falls_back_to_unanswerable() -> None:
    # Unrelated question: similarity stays below the threshold, so it is treated as unanswerable
    category, target_fields, gate = classify_question(
        "What is the capital city of France?", FACTS
    )
    assert category == "unanswerable"
    assert target_fields == []
    assert gate is False


def test_classify_age_question_via_embedding() -> None:
    # Age questions map to clinical_covariates.age
    category, target_fields, gate = classify_question(
        "How old is the patient?", FACTS
    )
    assert category == "retrieval"
    assert "clinical_covariates.age" in target_fields
    assert gate is True


def test_classify_should_i_trust_is_not_out_of_scope() -> None:
    # "should I trust ..." is a reasoning question, not a request for clinical action
    category, _target_fields, gate = classify_question(
        "How much should I trust this ranking given how spread out the saliency is?", FACTS
    )
    assert category != "out_of_scope"
    assert gate is True


def test_classify_question_debug_exposes_similarity_score() -> None:
    d = classify_question_debug("What treatment do you recommend for this patient?", FACTS)
    assert d["matched_by"] == "keyword"
    assert d["similarity"] is None

    d2 = classify_question_debug("What's this patient's MMSE score on file?", FACTS)
    assert d2["matched_by"] == "embedding"
    assert isinstance(d2["similarity"], float) and 0.0 <= d2["similarity"] <= 1.0
    assert d2["nearest_example"] is not None


# ===========================================================================
# [2] retrieve_fields
# ===========================================================================
def test_retrieve_fields_always_includes_argmax_class() -> None:
    # argmax_class is always included, even when no target fields are requested
    evidence = retrieve_fields([], FACTS, BASE_NARRATIVE)
    assert '"prediction_anchor.argmax_class": "CN"' in evidence["evidence_json"]


def test_retrieve_fields_extracts_target_field_values() -> None:
    evidence = retrieve_fields(["clinical_covariates.mmse_score"], FACTS, BASE_NARRATIVE)
    assert '"clinical_covariates.mmse_score": 29' in evidence["evidence_json"]
    # The keyword heuristic selects the base-narrative sentence that mentions MMSE
    assert "MMSE" in evidence["base_narrative_relevant"]


def test_retrieve_fields_unresolvable_field_is_silently_omitted() -> None:
    # A field without a value (cdr_score) is omitted rather than written as "null"
    evidence = retrieve_fields(["clinical_covariates.cdr_score"], FACTS, BASE_NARRATIVE)
    assert "cdr_score" not in evidence["evidence_json"]


# ===========================================================================
# [3] parse_answer (including the salvage path) & answer_schema_for_ollama
# ===========================================================================
def test_parse_answer_valid_json() -> None:
    raw = (
        '{"answer": "The patient\'s MMSE score is 29.", '
        '"cited_fields": ["clinical_covariates.mmse_score"], "answerable": true}'
    )
    out = parse_answer(raw)
    assert isinstance(out, AnswerOut)
    assert out.cited_fields == ["clinical_covariates.mmse_score"]
    assert out.answerable is True


def test_parse_answer_salvages_truncated_json() -> None:
    # Truncated JSON: answer and cited_fields are recovered, answerable defaults to False
    raw = '{"answer": "The MMSE score is 29.", "cited_fields": ["clinical_covariates.mmse_score'
    out = parse_answer(raw)
    assert out.answer == "The MMSE score is 29."
    assert out.answerable is False  # default value


def test_parse_answer_raises_when_answer_field_missing() -> None:
    # Without an "answer" key nothing can be recovered, so LLMOutputError is raised
    raw = '{"cited_fields": ["x"], "answerable": tru'
    try:
        parse_answer(raw)
        raise AssertionError("must raise LLMOutputError when the 'answer' field is entirely absent")
    except LLMOutputError:
        pass


def test_answer_schema_for_ollama_has_no_title_keys() -> None:
    # The schema carries no pydantic "title" keys
    schema = answer_schema_for_ollama()
    assert "title" not in schema
    assert set(schema["properties"]) == {"answer", "cited_fields", "answerable"}
    for prop in schema["properties"].values():
        assert "title" not in prop


# ===========================================================================
# [4]/[5] render_refusal -> run_qa_dag (a refusal always passes the DAG)
# ===========================================================================
def test_render_refusal_always_passes_full_qa_dag_out_of_scope() -> None:
    answer = render_refusal("out_of_scope", [], FACTS)
    assert answer.answerable is False
    assert answer.answer.strip() != REFUSAL_TOKEN  # a readable refusal, not the raw token
    report = run_qa_dag(answer, FACTS, BASE_NARRATIVE, "out_of_scope", source="deterministic_refusal")
    assert report.passed, "render_refusal must always pass run_qa_dag"


def test_render_refusal_always_passes_full_qa_dag_unanswerable() -> None:
    answer = render_refusal("unanswerable", ["clinical_covariates.mmse_score"], FACTS)
    report = run_qa_dag(answer, FACTS, BASE_NARRATIVE, "unanswerable", source="deterministic_refusal")
    assert report.passed


def test_render_refusal_restates_available_facts() -> None:
    # Resolvable target fields are restated and cited
    answer = render_refusal("unanswerable", ["clinical_covariates.mmse_score"], FACTS)
    assert "MMSE score is 29" in answer.answer
    assert answer.cited_fields == ["clinical_covariates.mmse_score"]


def test_render_refusal_omits_unresolvable_target_field() -> None:
    # A field without a value (cdr_score) is not restated
    answer = render_refusal("unanswerable", ["clinical_covariates.cdr_score"], FACTS)
    assert "CDR" not in answer.answer
    assert answer.cited_fields == []


# ===========================================================================
# audit_failure_reasons and qa_policy_sha
# ===========================================================================
def test_audit_failure_reasons_lists_only_failing_auditors() -> None:
    # Citing a non-existent field fails grounding (unresolvable_citation)
    bad = AnswerOut(
        answer="The tau PET SUVR is elevated.",
        cited_fields=["nonexistent.field"], answerable=True,
    )
    report = run_qa_dag(bad, FACTS, BASE_NARRATIVE, "retrieval", source="llm")
    assert not report.passed
    failing = audit_failure_reasons(report.to_dict())
    assert "grounding" in failing


def test_audit_failure_reasons_empty_when_guardrail_is_none() -> None:
    assert audit_failure_reasons(None) == []


# ===========================================================================
# Slice-percentage binding through the full `run_qa_dag()`. `top_slices` is a list of objects,
# so `audit_label_binding` does not cover it; `audit_structure_binding` must be part of the Q&A DAG.
# ===========================================================================
FACTS_QA_SLICES = FACTS.model_copy(deep=True)
FACTS_QA_SLICES.spatial_xai_anchor.top_slices = [
    TopSlice(name="Coronal", index=4, attention_share_pct=61.2),
    TopSlice(name="Coronal", index=5, attention_share_pct=24.8),
]
FACTS_QA_SLICES.spatial_xai_anchor.dominant_structures = ["hippocampus"]


def test_qa_dag_catches_slice_binding_swap() -> None:
    a = AnswerOut(
        answer="Coronal 4 (24.8%) and Coronal 5 (61.2%) show the highest attention share.",
        cited_fields=["spatial_xai_anchor.top_slices"], answerable=True,
    )
    report = run_qa_dag(a, FACTS_QA_SLICES, BASE_NARRATIVE, "retrieval", source="llm")
    assert not report.passed, (
        "swapped slice-name/percentage pairs in a Q&A answer must be rejected"
    )
    failing = audit_failure_reasons(report.to_dict())
    assert "structure_binding" in failing


def test_qa_dag_slice_binding_correct_passes() -> None:
    a = AnswerOut(
        answer="Coronal 4 (61.2%) and Coronal 5 (24.8%) show the highest attention share.",
        cited_fields=["spatial_xai_anchor.top_slices"], answerable=True,
    )
    report = run_qa_dag(a, FACTS_QA_SLICES, BASE_NARRATIVE, "retrieval", source="llm")
    assert report.passed


def test_qa_policy_sha_is_deterministic_and_present() -> None:
    sha1 = qa_policy_sha()
    sha2 = qa_policy_sha()
    assert sha1 == sha2
    assert sha1 != "missing", "docs/qa_scope_policy.md must exist"
    assert len(sha1) == 16


if __name__ == "__main__":
    test_classify_out_of_scope_keyword_gate()
    test_classify_retrieval_question_via_embedding()
    test_classify_leading_question_via_embedding()
    test_classify_unrelated_question_falls_back_to_unanswerable()
    test_classify_age_question_via_embedding()
    test_classify_should_i_trust_is_not_out_of_scope()
    test_classify_question_debug_exposes_similarity_score()
    test_retrieve_fields_always_includes_argmax_class()
    test_retrieve_fields_extracts_target_field_values()
    test_retrieve_fields_unresolvable_field_is_silently_omitted()
    test_parse_answer_valid_json()
    test_parse_answer_salvages_truncated_json()
    test_parse_answer_raises_when_answer_field_missing()
    test_answer_schema_for_ollama_has_no_title_keys()
    test_render_refusal_always_passes_full_qa_dag_out_of_scope()
    test_render_refusal_always_passes_full_qa_dag_unanswerable()
    test_render_refusal_restates_available_facts()
    test_render_refusal_omits_unresolvable_target_field()
    test_audit_failure_reasons_lists_only_failing_auditors()
    test_audit_failure_reasons_empty_when_guardrail_is_none()
    test_qa_policy_sha_is_deterministic_and_present()
    print("OK -- all qa_engine.py regressions (QA_GUARDRAIL_VERSION 0.6.0) passed.")
