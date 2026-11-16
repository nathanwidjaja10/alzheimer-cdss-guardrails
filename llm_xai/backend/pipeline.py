"""End-to-end orchestration of narrative generation and Q&A sessions.

Used by `api_router.py` (CDSS endpoints), `tools/evaluate_llm_paper.py` (paper benchmark) and
`notebooks/llm_xai_dev.ipynb`.

`run_narrative_pipeline` flow:
  predict_session -> build_fact_context -> cache lookup -> LLM draft -> run_dag
  -> on failure (variant c): regenerate once with corrections -> still failing: deterministic fallback
  -> optional FHIR bundle -> cache write
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from . import CONFIGS_DIR
from .guardrails import GuardrailReport, render_fallback, run_dag
from .llm_engine import (
    FactContext,
    LLMOutputError,
    NarrativeOut,
    SYSTEM_PROMPT,
    build_fact_context,
    cache_key,
    cache_read,
    cache_write,
    narrative_schema_for_ollama,
    ollama_chat,
    parse_narrative,
    render_user_prompt,
    run_manifest,
)

if TYPE_CHECKING:  # qa_engine imports from this module; see start_qa_session
    from .qa_engine import QASession, QATurn

MODELS_CFG_PATH = CONFIGS_DIR / "llm_models.yaml"
CLOUD_KINDS = {"openai", "google", "anthropic"}


@functools.lru_cache(maxsize=1)
def load_models_cfg(path: str | None = None) -> dict:
    """Load the model roster from llm_models.yaml (cached)."""
    p = Path(path) if path else MODELS_CFG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("models", {})


def _resolve_model(model_key: str, allow_cloud: bool, models_cfg: dict | None) -> dict:
    """Return the roster entry of a model; cloud models require explicit opt-in and an ethics clearance file."""
    cfg = (models_cfg or load_models_cfg())
    if model_key not in cfg:
        raise KeyError(f"model '{model_key}' does not exist in llm_models.yaml (available: {sorted(cfg)})")
    entry = dict(cfg[model_key])
    kind = entry.get("kind", "ollama")
    if kind in CLOUD_KINDS:
        clearance = (CONFIGS_DIR / "ethics_clearance.json").exists()
        if not (allow_cloud and clearance):
            raise PermissionError(
                f"model '{model_key}' is a cloud kind ({kind}). The cloud path is disabled. "
                f"Requires --allow-cloud AND llm_xai/configs/ethics_clearance.json."
            )
    elif kind != "ollama":
        raise ValueError(f"unknown kind for '{model_key}': {kind!r}")
    return entry


# ===========================================================================
# Core: generation from a fact context (skips `predict_session`, used by the evaluation harness)
# ===========================================================================
def generate_from_facts(
    facts: FactContext,
    model_key: str = "llama31_8b",
    variant: str = "c",
    *,
    enforce: bool | None = None,
    regenerate: bool = True,
    allow_cloud: bool = False,
    models_cfg: dict | None = None,
    seed_override: int | None = None,
) -> dict:
    """Generate a narrative and guardrail report for one (facts, model, variant).

    `enforce=None` enforces the guardrail (regeneration and fallback) only for variant 'c'; for
    variants 'a' and 'b' the guardrail is computed but does not alter the output.
    `seed_override` replaces the model's configured seed (used by the seed-stability check).
    """
    entry = _resolve_model(model_key, allow_cloud, models_cfg)
    if enforce is None:
        enforce = variant == "c"
    opts = dict(
        temperature=entry.get("temperature", 0.1),
        top_p=entry.get("top_p", 0.9),
        seed=seed_override if seed_override is not None else entry.get("seed", 42),
        num_predict=entry.get("num_predict", 1200),
    )
    # Variant a: loose JSON. Variants b and c: structured outputs (JSON schema)
    fmt = "json" if variant == "a" else narrative_schema_for_ollama()

    def _one(correction: str | None, source: str):
        # One generation attempt: render, call the LLM, parse, audit
        user = render_user_prompt(variant, facts, correction=correction)
        content, m = ollama_chat(entry["model"], SYSTEM_PROMPT, user, fmt=fmt, **opts)
        try:
            n = parse_narrative(content)
            return n, run_dag(n, facts, source=source), m, None
        except LLMOutputError as e:
            return None, None, m, str(e)

    narr, report, meta, parse_err = _one(None, "llm")
    n_calls = 1
    attempt1: dict | None = None
    attempt2: dict | None = None
    correction_used: str | None = None

    need_retry = enforce and regenerate and (narr is None or not report.passed)
    if need_retry:
        # Keep draft 1 and the correction so the caller can inspect the regeneration trail
        attempt1 = {
            "narrative": narr.model_dump() if narr is not None else None,
            "guardrail": report.to_dict() if report is not None else None,
            "parse_error": parse_err,
        }
        correction_used = report.summary() if report is not None else (
            "The previous reply was not valid JSON. Return STRICT JSON only, exactly the "
            "five required keys, nothing outside the object."
        )
        narr, report, meta, parse_err = _one(correction_used, "llm_regenerated")
        n_calls = 2
        attempt2 = {
            "narrative": narr.model_dump() if narr is not None else None,
            "guardrail": report.to_dict() if report is not None else None,
            "parse_error": parse_err,
        }

    # Deterministic fallback when the guardrail still fails
    if enforce and (narr is None or not report.passed):
        narr = NarrativeOut(**render_fallback(facts))
        report = run_dag(narr, facts, source="deterministic_fallback")
        parse_err = None

    return {
        "session_id": facts.session_id,
        "model": model_key,
        "variant": variant,
        "fact_digest": facts.fact_digest,
        "narrative": narr.model_dump() if narr is not None else None,
        "guardrail": report.to_dict() if report is not None else None,
        "parse_error": parse_err,
        "llm_meta": meta,
        "llm_calls": n_calls,
        "manifest": run_manifest(),
        # Regeneration trail (None when no retry happened)
        "attempt1": attempt1,
        "correction_used": correction_used,
        "attempt2": attempt2,
    }


# ===========================================================================
# From a session id (used by the CDSS endpoint and notebooks)
# ===========================================================================
def run_narrative_pipeline(
    session_id: str,
    model_key: str = "llama31_8b",
    variant: str = "c",
    *,
    use_cache: bool = True,
    build_fhir: bool = False,
    xai_method: str = "hirescam",
) -> dict:
    """Run classification, fact building and narrative generation for one session; optionally attach a FHIR bundle."""
    from backend.inference_engine import predict_session

    ai = predict_session(session_id, xai_method=xai_method)
    facts = build_fact_context(ai, session_id)

    key = cache_key(session_id, model_key, variant, facts.fact_digest)
    if use_cache:
        cached = cache_read(key)
        if cached is not None:
            return cached

    out = generate_from_facts(facts, model_key, variant)
    out["facts"] = facts.model_dump()

    g = out.get("guardrail") or {}
    if build_fhir and out.get("narrative") is not None:
        from .fhir_wrap import build_bundle_with_narrative

        cc = facts.clinical_covariates
        out["fhir_bundle"] = build_bundle_with_narrative(
            narrative=out["narrative"],
            provenance={
                "fact_digest": facts.fact_digest,
                "guardrail_status": "PASS" if g.get("passed") else "FAIL",
                "narrative_source": g.get("source", "n/a"),
                "model": model_key,
            },
            ai_result=ai,
            session_id=session_id,
            clinical={
                "age": cc.age,
                "gender": cc.gender,
                "mmse": cc.mmse_score if cc.mmse_available else None,
                "cdr": cc.cdr_score,
                "apoe": cc.apoe_genotype if cc.apoe_available else None,
            },
        )

    if use_cache:
        cache_write(key, out)
    return out


# ===========================================================================
# Q&A mode. qa_engine is imported inside the functions because it imports `_resolve_model`
# from this module (a top-level import on both sides would be circular).
# ===========================================================================
def start_qa_session(
    session_id: str,
    model_key: str = "llama31_8b",
    variant: str = "c",
    *,
    use_cache: bool = True,
) -> QASession:
    """Create a Q&A session: run (or reuse) the base narrative and pin it with the fact context.

    The only Q&A entry point that calls `predict_session`; later turns read the pinned facts only.
    """
    from .qa_engine import QASession

    out = run_narrative_pipeline(session_id, model_key=model_key, variant=variant, use_cache=use_cache)
    facts = FactContext(**out["facts"])
    base = NarrativeOut(**out["narrative"])
    return QASession(session_id=session_id, fact_context=facts, base_narrative=base, fact_digest=facts.fact_digest)


def ask(qa: QASession, question: str, **kwargs) -> QATurn:
    """Answer one question; `**kwargs` are forwarded to `qa_engine.answer_followup`."""
    from .qa_engine import answer_followup

    return answer_followup(qa, question, **kwargs)
