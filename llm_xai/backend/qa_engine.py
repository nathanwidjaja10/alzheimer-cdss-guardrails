"""Grounded Q&A mode on top of `llm_engine.py` and `guardrails.py`.

Single-turn flow (`answer_followup`):
  [1] classify_question  deterministic gate (keyword rules and embedding similarity)
  [2] retrieve_fields    fact-context slice and relevant base-narrative sentences
  [3] ollama_qa          controlled LLM answer (AnswerOut schema)
  [4] run_qa_dag         guardrail audit; on failure regenerate once, then fall back to a refusal
  [5] coherence_guard    the pinned diagnosis must not change; turn cap

It never calls `predict_session()`: only the fact context and base narrative pinned by
`pipeline.start_qa_session` are read.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from . import REPO_ROOT
from .guardrails import (
    MAX_QA_TURNS,
    CoherenceError,
    coherence_guard,
    render_refusal,
    run_qa_dag,
)
from .llm_engine import (
    FactContext,
    LLMOutputError,
    NarrativeOut,
    _env,
    _PROMPT_DIR,
    ollama_chat,
    resolve_fact_path,
    run_manifest,
)

# ===========================================================================
# Data models
# ===========================================================================
class AnswerOut(BaseModel):
    answer: str
    cited_fields: list[str] = []
    answerable: bool = False


class QATurn(BaseModel):
    question: str
    category: Literal["retrieval", "reasoning", "leading", "out_of_scope", "unanswerable"]
    answer_out: AnswerOut | None
    guardrail: dict | None = None          # GuardrailReport.to_dict() from run_qa_dag
    source: Literal["llm", "llm_regenerated", "deterministic_refusal"]
    llm_calls: int = 0
    wall_s: float = 0.0
    # Regeneration trail (None when no retry happened): the rejected first draft and the correction
    attempt1: dict | None = None
    correction_used: str | None = None
    attempt2: dict | None = None


class QASession(BaseModel):
    session_id: str
    fact_context: FactContext
    base_narrative: NarrativeOut
    fact_digest: str               # digest of the pinned facts
    turns: list[QATurn] = []


# ===========================================================================
# Example questions per category with their target fact fields. Keep in sync with
# docs/qa_scope_policy.md. `leading` questions are handled like `reasoning` (LLM plus guardrail).
# ===========================================================================
_CATEGORY_EXAMPLES: dict[str, list[tuple[str, list[str]]]] = {
    "retrieval": [
        ("Which imaging plane carries the most attention?",
         ["spatial_xai_anchor.primary_plane", "spatial_xai_anchor.plane_attention_share_pct"]),
        ("What is the out-of-domain Mahalanobis distance for this scan?",
         ["prediction_anchor.sq_mahalanobis_distance", "prediction_anchor.ood_threshold"]),
        ("What is the patient's MMSE score?",
         ["clinical_covariates.mmse_score"]),
        ("What percentage of saliency energy falls on the measured atrophy target?",
         ["spatial_xai_anchor.on_target_energy_pct", "spatial_xai_anchor.baseline_uniform_pct"]),
        ("What is the model's relative ranking for MCI?",
         ["prediction_anchor.relative_probabilities"]),
        ("What APOE genotype is recorded for this patient?",
         ["clinical_covariates.apoe_genotype"]),
        # Age/sex and alternative phrasings of out-of-domain and top-class questions
        ("What is this patient's age?", ["clinical_covariates.age"]),
        ("What is the patient's recorded sex?", ["clinical_covariates.gender"]),
        ("How far outside the training distribution does this scan fall?",
         ["prediction_anchor.sq_mahalanobis_distance", "prediction_anchor.ood_threshold"]),
        ("What is the model's top-ranked diagnostic class for this scan?",
         ["prediction_anchor.argmax_class", "prediction_anchor.relative_probabilities"]),
    ],
    "reasoning": [
        ("Why is the predicted class MCI and not AD?",
         ["prediction_anchor.argmax_class", "prediction_anchor.relative_probabilities"]),
        ("Does the MMSE score change the model's prediction?",
         ["clinical_covariates.mmse_score", "clinical_covariates.used_in_primary_decision"]),
        ("Why is this scan flagged as out-of-domain?",
         ["prediction_anchor.ood_status", "prediction_anchor.sq_mahalanobis_distance", "prediction_anchor.ood_threshold"]),
        ("What does the on-target saliency energy figure mean for how much to trust this ranking?",
         ["spatial_xai_anchor.on_target_energy_pct", "spatial_xai_anchor.ebpg_concentration_ratio"]),
        ("How reliable is this prediction given the out-of-domain flag?",
         ["prediction_anchor.ood_status", "prediction_anchor.calibration_note"]),
        ("Does the APOE4 allele support the model's prediction?",
         ["clinical_covariates.apoe_genotype", "clinical_covariates.covariate_caveat"]),
        # Genuine (non-leading) genotype question
        ("Would carrying the APOE4 allele make Alzheimer's a more likely explanation here?",
         ["clinical_covariates.apoe_genotype", "clinical_covariates.covariate_caveat"]),
    ],
    "leading": [
        ("Doesn't this scan basically confirm Alzheimer's?",
         ["prediction_anchor.argmax_class", "prediction_anchor.relative_probabilities"]),
        ("Isn't the atrophy pattern proof of AD?",
         ["spatial_xai_anchor.dominant_structures", "prediction_anchor.argmax_class"]),
        ("Given the APOE4 allele, shouldn't this really be AD?",
         ["clinical_covariates.apoe_genotype", "prediction_anchor.argmax_class"]),
        ("This looks just like Alzheimer's, right?",
         ["prediction_anchor.argmax_class"]),
        ("You're just being cautious calling it MCI -- it's really AD, isn't it?",
         ["prediction_anchor.argmax_class", "prediction_anchor.relative_probabilities"]),
    ],
    "out_of_scope": [
        ("Should we order a PET scan next?", []),
        ("What treatment do you recommend for this patient?", []),
        ("What is this patient's prognosis over the next two years?", []),
        ("Should the patient start donepezil?", []),
        ("Is this patient a candidate for a clinical trial?", []),
    ],
    "unanswerable": [
        ("What does the right temporal lobe show on the sagittal slice?", []),
        ("What was the patient's MMSE score a year ago?", []),
        ("What did the radiologist's report say about this scan?", []),
        ("What is the patient's family history of dementia?", []),
        ("What does the CSF amyloid/tau panel show?", []),
    ],
}

# Clinical-action requests. "should we/I/the patient" counts only when followed by a clinical-action verb.
_OUT_OF_SCOPE_KEYWORDS = re.compile(
    r"\b(?:recommend|treatment|therap\w*|medicat\w*|prognos\w*|follow-?up|next step|"
    r"refer(?:ral)?|biopsy|clinical trial|candidate for|donepezil|pet scan)\b"
    r"|\bshould (?:we|i|the patient) (?:order|start|begin|schedule|enroll|refer|prescribe|consider|try)\b",
    re.I,
)

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Minimum cosine similarity to the nearest example; below it the question is treated as unanswerable
_CLASSIFY_THRESHOLD = 0.40


@functools.lru_cache(maxsize=1)
def _embed_model():
    """Return the local sentence-embedding model, loaded lazily on first use."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_EMBED_MODEL_NAME)


@functools.lru_cache(maxsize=1)
def _category_bank():
    """Labels, questions, target fields and embeddings of all category examples (computed once)."""
    model = _embed_model()
    labels: list[str] = []
    questions: list[str] = []
    fields: list[list[str]] = []
    for cat, examples in _CATEGORY_EXAMPLES.items():
        for q, tf in examples:
            labels.append(cat)
            questions.append(q)
            fields.append(tf)
    embeddings = model.encode(questions, normalize_embeddings=True)
    return labels, questions, fields, embeddings


# ===========================================================================
# [1] Deterministic classification gate
# ===========================================================================
def classify_question_debug(q: str, fc: FactContext) -> dict:
    """Classify a question and also return the similarity score, nearest example and matching rule."""
    if _OUT_OF_SCOPE_KEYWORDS.search(q):
        return {
            "category": "out_of_scope", "target_fields": [], "answerable_gate": False,
            "similarity": None, "nearest_example": None, "matched_by": "keyword",
        }

    labels, questions, fields, embeddings = _category_bank()
    from sentence_transformers import util

    qv = _embed_model().encode([q], normalize_embeddings=True)
    sims = util.cos_sim(qv, embeddings)[0].tolist()
    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    best_sim = sims[best_idx]

    if best_sim < _CLASSIFY_THRESHOLD:
        return {
            "category": "unanswerable", "target_fields": [], "answerable_gate": False,
            "similarity": best_sim, "nearest_example": questions[best_idx], "matched_by": "below_threshold",
        }

    category = labels[best_idx]
    return {
        "category": category, "target_fields": fields[best_idx],
        "answerable_gate": category not in ("out_of_scope", "unanswerable"),
        "similarity": best_sim, "nearest_example": questions[best_idx], "matched_by": "embedding",
    }


def classify_question(q: str, fc: FactContext) -> tuple[str, list[str], bool]:
    """Return (category, target_fields, answerable_gate) for a question.

    Keyword rules catch clear out-of-scope requests; otherwise the category and target fields come
    from the nearest example by embedding similarity. Below the similarity threshold the question
    is treated as unanswerable. `fc` is accepted for future session-aware gating and is unused.
    """
    d = classify_question_debug(q, fc)
    return d["category"], d["target_fields"], d["answerable_gate"]


# ===========================================================================
# [2] Retrieval from the session's structured facts (no external corpus)
# ===========================================================================
def _relevant_narrative_sentences(base: NarrativeOut, target_fields: list[str], limit: int = 2) -> list[str]:
    """Return up to `limit` base-narrative sentences that mention a keyword of the target fields."""
    keywords = {tf.rsplit(".", 1)[-1].split("_")[0].lower() for tf in target_fields}
    keywords = {k for k in keywords if len(k) > 2}
    if not keywords:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", base.as_text())
    hits = [s.strip() for s in sentences if any(k in s.lower() for k in keywords)]
    return hits[:limit]


def retrieve_fields(target_fields: list[str], fc: FactContext, base: NarrativeOut) -> dict:
    """Return {'evidence_json': fact subset, 'base_narrative_relevant': up to 2 sentences}.

    `prediction_anchor.argmax_class` is always included so the model sees the decided class.
    """
    fields = list(dict.fromkeys(["prediction_anchor.argmax_class", *target_fields]))
    evidence: dict[str, Any] = {}
    for tf in fields:
        val = resolve_fact_path(fc, tf)
        if val is not None:
            evidence[tf] = val
    sentences = _relevant_narrative_sentences(base, target_fields)
    return {
        "evidence_json": json.dumps(evidence, indent=2, default=str),
        "base_narrative_relevant": "\n".join(sentences) if sentences else "(none directly relevant)",
    }


# ===========================================================================
# [3] Controlled LLM answer
# ===========================================================================
QA_SYSTEM_PROMPT = (_PROMPT_DIR / "qa_system_v1.txt").read_text(encoding="utf-8")

_ANSWER_SCHEMA_HINT = json.dumps(
    {
        "answer": "1-3 sentences, or exactly \"<REFUSE>\" if unanswerable/out of scope",
        "cited_fields": "list of the exact Evidence keys you relied on",
        "answerable": "bool",
    },
    indent=2,
)


def answer_schema_for_ollama() -> dict:
    """JSON schema of `AnswerOut` for Ollama structured outputs."""
    s = AnswerOut.model_json_schema()
    s.pop("title", None)
    for p in s.get("properties", {}).values():
        p.pop("title", None)
    return s


def render_qa_user_prompt(question: str, evidence: dict, correction: str | None = None) -> str:
    """Render the Q&A user prompt from the question, evidence and optional correction feedback."""
    tpl = _env.get_template("qa_user.jinja")
    return tpl.render(
        evidence_json=evidence["evidence_json"],
        base_narrative_relevant=evidence["base_narrative_relevant"],
        question=question,
        schema=_ANSWER_SCHEMA_HINT,
        correction=correction,
    )


_ANSWER_KEYS = ("answer",)  # only 'answer' is required; the other fields have defaults


def _salvage_answer_fields(content: str) -> dict:
    """Extract answer fields by regex from malformed or truncated JSON."""
    out: dict[str, Any] = {}
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    if m:
        s = m.group(1)
        if "\\" in s:
            try:
                s = json.loads(f'"{s}"')
            except json.JSONDecodeError:
                pass
        out["answer"] = s
    m2 = re.search(r'"cited_fields"\s*:\s*\[(.*?)\]', content, re.DOTALL)
    if m2:
        out["cited_fields"] = re.findall(r'"([^"]+)"', m2.group(1))
    m3 = re.search(r'"answerable"\s*:\s*(true|false)', content, re.I)
    if m3:
        out["answerable"] = m3.group(1).lower() == "true"
    return out


def parse_answer(content: str) -> AnswerOut:
    """Parse the LLM output into an AnswerOut, recovering from stray text or malformed JSON when possible."""
    data: dict | None = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        data = _salvage_answer_fields(content)
        if set(_ANSWER_KEYS) - set(data):
            raise LLMOutputError(f"LLM JSON could not be parsed & the 'answer' field is missing: got {sorted(data)}")
    try:
        return AnswerOut(**data)
    except (ValidationError, TypeError) as e:
        raise LLMOutputError(f"LLM JSON does not match the AnswerOut schema: {e}") from e


def ollama_qa(question: str, evidence: dict, model: str, *, correction: str | None = None,
              temperature: float = 0.1, top_p: float = 0.9, seed: int = 42,
              num_predict: int = 400) -> tuple[AnswerOut, dict]:
    """Render the Q&A prompt, call the LLM with the AnswerOut schema and parse the reply; returns (answer, metadata)."""
    user = render_qa_user_prompt(question, evidence, correction)
    content, meta = ollama_chat(
        model, QA_SYSTEM_PROMPT, user, fmt=answer_schema_for_ollama(),
        temperature=temperature, top_p=top_p, seed=seed, num_predict=num_predict,
    )
    return parse_answer(content), meta


# ===========================================================================
# Single-turn orchestration
# ===========================================================================
def answer_followup(qa: QASession, question: str, model_key: str = "llama31_8b", *,
                     regenerate: bool = True, allow_cloud: bool = False,
                     models_cfg: dict | None = None) -> QATurn:
    """Answer one question and append the turn to the session.

    Out-of-scope and unanswerable questions get a deterministic refusal without calling the LLM.
    Otherwise the LLM answer is audited by `run_qa_dag`; on failure it is regenerated once with
    the correction feedback, and a still failing answer is replaced by a refusal. An answer that
    passes is also checked by `coherence_guard` and replaced by a refusal if it would shift the
    diagnosis.
    """
    # Local import avoids a circular import (pipeline imports qa_engine)
    from .pipeline import _resolve_model

    if len(qa.turns) >= MAX_QA_TURNS:
        raise CoherenceError(f"QA session is already {len(qa.turns)} turns (limit {MAX_QA_TURNS}); start a new session.")

    t0 = time.time()
    category, target_fields, _gate = classify_question(question, qa.fact_context)

    if category in ("out_of_scope", "unanswerable"):
        answer = render_refusal(category, target_fields, qa.fact_context)
        report = run_qa_dag(answer, qa.fact_context, qa.base_narrative, category, source="deterministic_refusal")
        turn = QATurn(question=question, category=category, answer_out=answer,
                      guardrail=report.to_dict(), source="deterministic_refusal",
                      llm_calls=0, wall_s=round(time.time() - t0, 3))
        qa.turns.append(turn)
        return turn

    entry = _resolve_model(model_key, allow_cloud, models_cfg)
    opts = dict(temperature=entry.get("temperature", 0.1), top_p=entry.get("top_p", 0.9),
                seed=entry.get("seed", 42), num_predict=400)
    evidence = retrieve_fields(target_fields, qa.fact_context, qa.base_narrative)

    answer, _meta = ollama_qa(question, evidence, entry["model"], **opts)
    report = run_qa_dag(answer, qa.fact_context, qa.base_narrative, category, source="llm")
    n_calls, source = 1, "llm"
    # Regeneration trail (None when the first draft passes)
    attempt1: dict | None = None
    correction_used: str | None = None
    attempt2: dict | None = None

    if not report.passed and regenerate:
        attempt1 = {"answer": answer.model_dump(), "guardrail": report.to_dict()}
        correction_used = report.summary()
        answer2, _meta2 = ollama_qa(question, evidence, entry["model"], correction=correction_used, **opts)
        report2 = run_qa_dag(answer2, qa.fact_context, qa.base_narrative, category, source="llm_regenerated")
        n_calls = 2
        attempt2 = {"answer": answer2.model_dump(), "guardrail": report2.to_dict()}
        if report2.passed:
            answer, report, source = answer2, report2, "llm_regenerated"
        else:
            answer = render_refusal(category, target_fields, qa.fact_context)
            report = run_qa_dag(answer, qa.fact_context, qa.base_narrative, category, source="deterministic_refusal")
            source = "deterministic_refusal"
    elif not report.passed:
        attempt1 = {"answer": answer.model_dump(), "guardrail": report.to_dict()}
        answer = render_refusal(category, target_fields, qa.fact_context)
        report = run_qa_dag(answer, qa.fact_context, qa.base_narrative, category, source="deterministic_refusal")
        source = "deterministic_refusal"

    turn = QATurn(question=question, category=category, answer_out=answer,
                  guardrail=report.to_dict(), source=source, llm_calls=n_calls,
                  wall_s=round(time.time() - t0, 3), attempt1=attempt1,
                  correction_used=correction_used, attempt2=attempt2)

    if source != "deterministic_refusal":
        try:
            coherence_guard(turn, qa)
        except CoherenceError:
            refusal = render_refusal("out_of_scope", [], qa.fact_context)
            refusal_report = run_qa_dag(refusal, qa.fact_context, qa.base_narrative,
                                        "out_of_scope", source="deterministic_refusal")
            turn = QATurn(question=question, category=category, answer_out=refusal,
                          guardrail=refusal_report.to_dict(), source="deterministic_refusal",
                          llm_calls=n_calls, wall_s=round(time.time() - t0, 3),
                          attempt1=attempt1, correction_used=correction_used, attempt2=attempt2)

    qa.turns.append(turn)
    return turn


def audit_failure_reasons(guardrail: dict | None) -> list[str]:
    """Names of the auditors that failed in a `GuardrailReport.to_dict()` (empty when none failed)."""
    if not guardrail:
        return []
    return [a["name"] for a in guardrail.get("audits", []) if not a["passed"]]


# ===========================================================================
# Reproducibility
# ===========================================================================
def qa_policy_sha() -> str:
    """Short hash of `docs/qa_scope_policy.md`, recorded in the run manifest."""
    p = REPO_ROOT / "llm_xai" / "docs" / "qa_scope_policy.md"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"


def qa_run_manifest() -> dict:
    """Run manifest extended with the Q&A policy, classifier, NLI and turn-cap settings."""
    from .guardrails import _NLI_ENTAILMENT_THRESHOLD, _NLI_MODEL_NAME

    m = run_manifest()
    m.update({
        "qa_policy_sha": qa_policy_sha(),
        "qa_classify_model": _EMBED_MODEL_NAME,
        "qa_classify_threshold": _CLASSIFY_THRESHOLD,
        "qa_nli_model": _NLI_MODEL_NAME,
        "qa_nli_threshold": _NLI_ENTAILMENT_THRESHOLD,
        "qa_max_turns": MAX_QA_TURNS,
    })
    return m
