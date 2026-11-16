"""Fact context builder, schemas, local LLM client, prompt rendering and narrative cache.

Contents:
  - Pydantic schemas: FactContext (and sub-models) and NarrativeOut
  - build_fact_context(ai_result, session_id) -> FactContext
  - ollama_chat(...) and parse_narrative(...): local LLM inference through Ollama
  - render_user_prompt(variant, facts, correction): Jinja2 prompt templates
  - cache_read / cache_write: on-disk narrative cache

All inference is local; no cloud model API is called.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from . import GUARDRAIL_VERSION, REPO_ROOT, RESULTS_DIR, SCHEMA_VERSION
from .volebpg_aggregate import aggregate_plane_attention, aggregate_volebpg

# ===========================================================================
# Text constants used in the fact context, fallback narrative and prompt
# ===========================================================================
CALIBRATION_NOTE = (
    "Temperature scaling did not generalise (T~0); the probability figures are a "
    "RELATIVE RANKING CN<MCI<AD, not absolute confidence percentages."
)
COVARIATE_CAVEAT = (
    "Covariates are read-only (used_in_primary_decision=false). APOE genotype is "
    "encoded as a single e4-presence flag, without accounting for the protective e2 "
    "allele, and must never be presented as evidence for or against Alzheimer's "
    "disease — regardless of which alleles are present."
)
STRUCTURE_SALIENCY_CAVEAT = (
    "Saliency concentration does not by itself confirm the disease-specific atrophy "
    "pattern reported in the literature; ventricular dominance and non-monotonic "
    "staging across CN/MCI/AD have been observed independently of diagnostic accuracy."
)
MEASURED_STRUCTURES = [
    "hippocampus", "lateral ventricle", "inferior lateral ventricle",
    "third ventricle", "amygdala", "cerebral cortex",
]
# Structure keys of `results/aga_raw/*.json` (compute_structure_saliency.py) -> whitelist terms
_AGA_STRUCTURE_DISPLAY_NAME = {
    "hippocampus": "hippocampus",
    "ventricles": "lateral ventricle",
    "amygdala": "amygdala",
}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
CACHE_DIR = REPO_ROOT / "llm_xai" / "results" / "narratives"


# ===========================================================================
# Pydantic schemas of the fact context
# ===========================================================================
class PredictionAnchor(BaseModel):
    argmax_class: Literal["CN", "MCI", "AD"]
    probability_semantics: str = "relative_ranking_only"
    relative_probabilities: dict[str, float]
    calibration_note: str = CALIBRATION_NOTE
    decision_rule: str = "argmax (vision-only; covariates masked)"
    ood_status: Literal["in_validated_domain", "outside_validated_domain", "unavailable"]
    sq_mahalanobis_distance: float | None = None
    ood_threshold: float | None = None
    validated_domain_note: str | None = None


class TopSlice(BaseModel):
    name: str | None
    index: int
    attention_share_pct: float
    on_target_pct: float | None = None


class SpatialXaiAnchor(BaseModel):
    xai_method: str = "hirescam"
    primary_plane: str | None = None
    plane_attention_share_pct: dict[str, float] = Field(default_factory=dict)
    top_slices: list[TopSlice] = Field(default_factory=list)
    ebpg_measured_structures: list[str] = Field(default_factory=lambda: list(MEASURED_STRUCTURES))
    on_target_energy_pct: float | None = None
    baseline_uniform_pct: float | None = None
    ebpg_concentration_ratio: float | None = None
    dominant_structures: list[str] = Field(default_factory=list)
    aseg_available: bool = False
    structure_saliency_caveat: str = STRUCTURE_SALIENCY_CAVEAT


class ClinicalCovariates(BaseModel):
    used_in_primary_decision: bool = False
    age: float | int | None = None
    gender: str | None = None
    mmse_score: float | int | None = None
    mmse_available: bool = False
    cdr_score: float | None = None
    apoe_genotype: str | None = None
    apoe_available: bool = False
    covariate_caveat: str = COVARIATE_CAVEAT


class ModelProvenance(BaseModel):
    name: str = "MobileViT25DMultimodalV2-Unified"
    checkpoint_loaded: bool | None = None
    internal_test_macro_f1: float | None = None
    external_adni1_macro_f1: float | None = None


class FactContext(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    prediction_anchor: PredictionAnchor
    spatial_xai_anchor: SpatialXaiAnchor
    clinical_covariates: ClinicalCovariates
    model_provenance: ModelProvenance = Field(default_factory=ModelProvenance)
    fact_digest: str = ""

    def compute_digest(self) -> str:
        payload = self.model_dump(exclude={"fact_digest"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def finalized(self) -> "FactContext":
        self.fact_digest = self.compute_digest()
        return self

    def to_prompt_json(self) -> str:
        """Return the fact context as the JSON embedded in the prompt.

        Omitted from the prompt (but kept on the object and in the digest):
          - model_provenance and fact_digest, so the LLM does not cite model metrics as its own;
          - the calibration, covariate and structure-saliency caveats, which are long
            instructional sentences already stated in the system prompt.
        """
        d = self.model_dump(exclude={"model_provenance", "fact_digest"}, mode="json")
        d.get("prediction_anchor", {}).pop("calibration_note", None)
        d.get("clinical_covariates", {}).pop("covariate_caveat", None)
        d.get("spatial_xai_anchor", {}).pop("structure_saliency_caveat", None)
        return json.dumps(d, indent=2, default=str)


def resolve_fact_path(fc: FactContext, path: str) -> Any | None:
    """Return the value at a dotted fact path (e.g. `clinical_covariates.mmse_score`), or None.

    Shared by `qa_engine.py` and `guardrails.py` to resolve `target_fields` and `cited_fields`.
    """
    node: Any = fc.model_dump(mode="json")
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


class NarrativeOut(BaseModel):
    """Schema of the narrative returned by the LLM (closed JSON)."""

    visual_findings: str
    multimodal_correlation: str
    differential_justification: str
    differential_conclusion: Literal["CN", "MCI", "AD"]
    uncertainty_note: str

    def as_text(self) -> str:
        """Concatenation of the free-text sections (excluding the class label) for guardrail auditing."""
        return "\n".join(
            [self.visual_findings, self.multimodal_correlation,
             self.differential_justification, self.uncertainty_note]
        )


NARRATIVE_SCHEMA_HINT = json.dumps(
    {
        "visual_findings": "1-3 sentences localising the saliency in the image",
        "multimodal_correlation": "1-2 sentences on the cognitive/genetic context (covariates are read-only; APOE genotype is not evidence for AD, whether or not it includes the e4 allele)",
        "differential_justification": "why this class rather than the alternatives",
        "differential_conclusion": "CN|MCI|AD (MUST equal argmax_class)",
        "uncertainty_note": "out-of-domain / validated-range caveat",
    },
    indent=2,
)


# ===========================================================================
# Clinical metadata (the CDR is not part of the `predict_session` output)
# ===========================================================================
_CLIN: dict[str, Any] | None = None


def _load_clinical() -> dict[str, Any]:
    """Load and cache the clinical metadata of both cohorts."""
    global _CLIN
    if _CLIN is None:
        _CLIN = {}
        for p in (REPO_ROOT / "oasis_data" / "clinical_metadata.json",
                  REPO_ROOT / "adni_data" / "clinical_metadata.json",
                  REPO_ROOT / "dummy_data" / "clinical_metadata.json"):
            if p.exists():
                try:
                    _CLIN.update(json.loads(p.read_text()))
                except Exception:  # noqa: BLE001
                    pass
    return _CLIN


def normalize_apoe(raw: Any) -> str | None:
    """Convert a raw APOE genotype string ('34') to the 'E3/E4' form, or None when missing.

    The two single digits are structural numbers that the numeric auditor skips.
    """
    if raw in (None, "", "#N/A", "0", "nan", "None"):
        return None
    digits = re.sub(r"[^0-9]", "", str(raw))
    if len(digits) == 2 and digits.isdigit():
        return f"E{digits[0]}/E{digits[1]}"
    return str(raw)


def dominant_structures_from_aga(session_id: str, fallback: list[str]) -> list[str]:
    """Return the structure with the highest attribution energy for a session.

    Reads `results/aga_raw/<session_id>.json` (compute_structure_saliency.py); `fallback` is
    returned when the file is missing or unreadable.
    """
    p = RESULTS_DIR / "aga_raw" / f"{session_id}.json"
    if not p.exists():
        return fallback
    try:
        top = json.loads(p.read_text()).get("top_structure_by_energy")
    except Exception:  # noqa: BLE001
        return fallback
    name = _AGA_STRUCTURE_DISPLAY_NAME.get(top)
    return [name] if name else fallback


def read_cdr(session_id: str, canonical_id: str) -> float | None:
    """Return the session's global CDR from the clinical metadata, or None."""
    c = _load_clinical()
    for k in (canonical_id, session_id, session_id.replace("_MR_", "_")):
        if k in c:
            try:
                return float(c[k].get("cdr"))
            except (TypeError, ValueError):
                return None
    return None


_TRAIN_META: dict | None = None


def _upstream_f1() -> tuple[float | None, float | None]:
    """Return (internal test macro-F1, external ADNI macro-F1) for the model provenance.

    The internal value is read from checkpoints/training_metadata_v2_unified.json; the external
    value is a constant.
    """
    global _TRAIN_META
    if _TRAIN_META is None:
        p = REPO_ROOT / "checkpoints" / "training_metadata_v2_unified.json"
        try:
            _TRAIN_META = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            _TRAIN_META = {}
    ev = _TRAIN_META.get("evaluation", {})
    internal = ev.get("internal_test_macro_f1_argmax")
    internal = round(internal / 100.0, 4) if isinstance(internal, (int, float)) else None
    return internal, 0.4059  # external ADNI macro-F1


# Standard wording of the validated-range note. A note that references notebook cells is
# replaced by it so that nothing notebook-specific reaches the prompt.
_VALIDATED_DOMAIN_NOTE = (
    "The model is validated only for OASIS-3 structural MRI processed with FreeSurfer. "
    "Scans from other sites or pipelines fall outside the validated range and are "
    "flagged by the out-of-domain check."
)


def map_ood_status(ood: dict | None) -> dict:
    """Map the engine's out-of-domain result to the fact-context fields."""
    if not ood or not ood.get("available"):
        return {"ood_status": "unavailable"}
    raw_note = str(ood.get("validated_domain_note") or "")
    note = _VALIDATED_DOMAIN_NOTE if (not raw_note or re.search(r"cell\s*\d", raw_note, re.I)) else raw_note
    return {
        "ood_status": "in_validated_domain" if ood.get("in_validated_domain") else "outside_validated_domain",
        "sq_mahalanobis_distance": ood.get("distance"),
        "ood_threshold": ood.get("threshold"),
        "validated_domain_note": note,
    }


# ===========================================================================
# Fact builder
# ===========================================================================
def build_fact_context(ai_result: dict, session_id: str) -> FactContext:
    """Assemble the fact context from the output of `predict_session()`."""
    canonical = ai_result.get("canonical_session_id", session_id)
    probs = {k: round(float(v), 4) for k, v in ai_result["probabilities"].items()}
    attn = ai_result.get("attention_weights", [])
    heat = ai_result.get("slice_heatmaps", [])

    plane = aggregate_plane_attention(attn)
    vol = aggregate_volebpg(session_id, canonical, heat, attn)
    ood = map_ood_status(ai_result.get("out_of_domain", {}))

    on_target_by_idx = {int(s["index"]): s.get("on_target_pct") for s in heat}
    top_slices = [
        TopSlice(
            name=t.get("name"),
            index=int(t.get("index", -1)),
            attention_share_pct=float(t.get("attention_share_pct", 0.0)),
            on_target_pct=on_target_by_idx.get(int(t.get("index", -1))),
        )
        for t in plane["top_slices"]
    ]

    mm = ai_result.get("multimodal_inputs_used", {})
    mi = ai_result.get("model_info", {})

    fc = FactContext(
        session_id=session_id,
        prediction_anchor=PredictionAnchor(
            argmax_class=ai_result["diagnosis"],
            relative_probabilities=probs,
            decision_rule=f'{ai_result.get("decision_rule_active", "argmax")} (vision-only; covariates masked)',
            ood_status=ood["ood_status"],
            sq_mahalanobis_distance=ood.get("sq_mahalanobis_distance"),
            ood_threshold=ood.get("ood_threshold"),
            validated_domain_note=ood.get("validated_domain_note"),
        ),
        spatial_xai_anchor=SpatialXaiAnchor(
            xai_method=ai_result.get("xai_method", "hirescam"),
            primary_plane=plane["primary_plane"],
            plane_attention_share_pct=plane["plane_attention_share_pct"],
            top_slices=top_slices,
            on_target_energy_pct=vol["on_target_energy_pct"],
            baseline_uniform_pct=vol["baseline_uniform_pct"],
            ebpg_concentration_ratio=vol["ebpg_concentration_ratio"],
            dominant_structures=dominant_structures_from_aga(session_id, vol["dominant_structures"]),
            aseg_available=vol["aseg_available"],
        ),
        clinical_covariates=ClinicalCovariates(
            age=mm.get("age"),
            gender=mm.get("gender"),
            mmse_score=mm.get("mmse_score"),
            mmse_available=bool(mm.get("mmse_available")),
            cdr_score=read_cdr(session_id, canonical),
            apoe_genotype=normalize_apoe(mm.get("apoe_genotype")),
            apoe_available=bool(mm.get("apoe_available")),
        ),
        model_provenance=ModelProvenance(
            name=mi.get("name", "MobileViT25DMultimodalV2-Unified"),
            checkpoint_loaded=mi.get("checkpoint_loaded"),
            internal_test_macro_f1=_upstream_f1()[0],
            external_adni1_macro_f1=_upstream_f1()[1],
        ),
    ).finalized()
    return fc


def minimize_for_external(facts: FactContext) -> dict:
    """Return a minimal payload for an external model: hashed session id and no provenance.

    Not used while inference is local.
    """
    salt = os.environ.get("LLMXAI_HASH_SALT", "local-salt")
    d = facts.model_dump(exclude={"model_provenance"})
    d["session_id"] = "sha256:" + hashlib.sha256((salt + facts.session_id).encode()).hexdigest()[:16]
    d["fact_digest"] = facts.fact_digest
    return d


# ===========================================================================
# Prompt (Jinja2)
# ===========================================================================
_env = Environment(
    loader=FileSystemLoader(str(_PROMPT_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
SYSTEM_PROMPT = (_PROMPT_DIR / "system_v1.txt").read_text(encoding="utf-8")
VARIANT_TEMPLATES = {
    "a": "variant_a_freeform.jinja",
    "b": "variant_b_schema.jinja",
    "c": "variant_c_guarded.jinja",
}


def prompt_sha() -> str:
    """Short hash of the system prompt and variant templates, used for cache keys and reports."""
    h = hashlib.sha256()
    h.update(SYSTEM_PROMPT.encode("utf-8"))
    for f in sorted(_PROMPT_DIR.glob("variant_*.jinja")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def render_user_prompt(variant: str, facts: FactContext, correction: str | None = None) -> str:
    """Render the user prompt of variant a, b or c, with optional correction feedback."""
    if variant not in VARIANT_TEMPLATES:
        raise ValueError(f"unknown variant: {variant!r} (choose a|b|c)")
    tpl = _env.get_template(VARIANT_TEMPLATES[variant])
    return tpl.render(
        fact_json=facts.to_prompt_json(),
        schema=NARRATIVE_SCHEMA_HINT,
        argmax=facts.prediction_anchor.argmax_class,   # value required for `differential_conclusion`
        correction=correction,
    )


# ===========================================================================
# Ollama client (local only)
# ===========================================================================
class LLMError(RuntimeError):
    pass


class LLMOutputError(LLMError):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(LLMError),
    reraise=True,
)
def narrative_schema_for_ollama() -> dict:
    """JSON schema of `NarrativeOut` for Ollama structured outputs (variants B and C).

    Variant A uses loose `format="json"` instead.
    """
    s = NarrativeOut.model_json_schema()
    s.pop("title", None)
    for p in s.get("properties", {}).values():
        p.pop("title", None)
    return s


def ollama_chat(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    top_p: float = 0.9,
    seed: int = 42,
    num_predict: int = 1200,
    fmt: str | dict | None = "json",
    stop: list[str] | None = None,
    timeout: float = 300.0,
) -> tuple[str, dict]:
    """Call `POST {OLLAMA_HOST}/api/chat` and return (content, metadata).

    `fmt` is the string "json" or a JSON schema (structured outputs).
    """
    options: dict[str, Any] = {"temperature": temperature, "top_p": top_p, "seed": seed, "num_predict": num_predict}
    if stop:
        options["stop"] = stop
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": options,
    }
    if fmt:
        payload["format"] = fmt
    try:
        r = httpx.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as e:  # connection failure or non-2xx status
        raise LLMError(f"Ollama call failed (model={model}): {e}") from e

    body = r.json()
    content = (body.get("message") or {}).get("content", "") or ""
    meta = {
        "model": model,
        "eval_count": body.get("eval_count"),
        "eval_duration_ns": body.get("eval_duration"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "total_duration_ns": body.get("total_duration"),
        "load_duration_ns": body.get("load_duration"),
    }
    return content, meta


_NARR_KEYS = (
    "visual_findings", "multimodal_correlation", "differential_justification",
    "differential_conclusion", "uncertainty_note",
)


def _salvage_fields(content: str) -> dict:
    """Extract the five narrative fields by regex from malformed or truncated JSON."""
    out: dict[str, str] = {}
    for k in _NARR_KEYS:
        m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
        if m:
            s = m.group(1)
            if "\\" in s:
                try:
                    s = json.loads(f'"{s}"')
                except json.JSONDecodeError:
                    pass
            out[k] = s
    return out


def parse_narrative(content: str) -> NarrativeOut:
    """Parse the LLM output into a NarrativeOut, recovering from stray text or malformed JSON when possible."""
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
        data = _salvage_fields(content)
        if set(_NARR_KEYS) - set(data):
            raise LLMOutputError(
                f"LLM JSON could not be parsed & fields are incomplete: got {sorted(data)}"
            )
    try:
        return NarrativeOut(**data)
    except (ValidationError, TypeError) as e:
        raise LLMOutputError(f"LLM JSON does not match the NarrativeOut schema: {e}") from e


# ===========================================================================
# On-disk narrative cache
# ===========================================================================
def cache_key(session_id: str, model: str, variant: str, fact_digest: str) -> str:
    """Cache key over everything that determines a narrative (inputs, prompts, guardrail version)."""
    raw = f"{session_id}|{model}|{variant}|{fact_digest}|{prompt_sha()}|{GUARDRAIL_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def cache_read(key: str) -> dict | None:
    """Return the cached result for a key, or None."""
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


def cache_write(key: str, obj: dict) -> None:
    """Store a result under a key; failures are logged and ignored."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (CACHE_DIR / f"{key}.json").write_text(json.dumps(obj, default=str, indent=2))
    except Exception as e:  # noqa: BLE001
        print(f"[llm_engine] failed to write cache {key}: {e}")


def run_manifest() -> dict:
    """Versions and settings recorded in every report for reproducibility."""
    return {
        "schema_version": SCHEMA_VERSION,
        "guardrail_version": GUARDRAIL_VERSION,
        "prompt_sha": prompt_sha(),
        "ollama_host": OLLAMA_HOST,
    }
