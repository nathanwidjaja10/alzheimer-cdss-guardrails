"""HTTP endpoints for narrative generation and Q&A, mounted optionally by `backend/app.py`."""
from __future__ import annotations

import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # root repo

from fastapi import APIRouter, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from llm_xai.backend.guardrails import CoherenceError  # noqa: E402
from llm_xai.backend.pipeline import ask, run_narrative_pipeline, start_qa_session  # noqa: E402
from llm_xai.backend.qa_engine import QASession  # noqa: E402

router = APIRouter(tags=["llm-xai"])

# In-memory Q&A sessions keyed by token. They are lost on restart and are not shared across
# worker processes.
_QA_SESSIONS: dict[str, QASession] = {}


@router.post("/api/xai/narrative/{session_id}")
def generate_narrative(
    session_id: str,
    model: str = "llama31_8b",
    variant: str = "c",
    include_fhir: bool = True,
    use_cache: bool = True,
):
    """Generate a guardrail-audited narrative for one session.

    The audited narrative is embedded in the FHIR DiagnosticReport unless include_fhir is False.
    Response: {narrative, guardrail, facts, fact_digest, llm_meta, [fhir_bundle]}.
    """
    if variant not in ("a", "b", "c"):
        raise HTTPException(400, f"invalid variant: {variant!r} (a|b|c)")
    try:
        return run_narrative_pipeline(
            session_id,
            model_key=model,
            variant=variant,
            use_cache=use_cache,
            build_fhir=include_fhir,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except PermissionError as e:  # a cloud model was requested without clearance
        raise HTTPException(403, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"narrative pipeline error: {e}")


# ===========================================================================
# Q&A mode: create a session (pins the facts and base narrative), then ask one turn per request
# ===========================================================================
class QARequest(BaseModel):
    question: str


@router.post("/api/xai/narrative/{session_id}/session")
def create_qa_session(
    session_id: str,
    model: str = "llama31_8b",
    variant: str = "c",
    use_cache: bool = True,
):
    """Create a Q&A session: run (or reuse) the base narrative and pin it with the fact context.

    Response: {qa_token, base_narrative, fact_digest}.
    """
    if variant not in ("a", "b", "c"):
        raise HTTPException(400, f"invalid variant: {variant!r} (a|b|c)")
    try:
        qa = start_qa_session(session_id, model_key=model, variant=variant, use_cache=use_cache)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"qa session error: {e}")

    token = secrets.token_hex(16)
    _QA_SESSIONS[token] = qa
    return {
        "qa_token": token,
        "base_narrative": qa.base_narrative.model_dump(),
        "fact_digest": qa.fact_digest,
    }


@router.post("/api/xai/qa/{qa_token}")
def ask_qa(qa_token: str, body: QARequest):
    """Answer one question in an existing Q&A session using only its pinned facts.

    Response: {answer, cited_fields, answerable, category, guardrail, source, llm_calls, wall_s}.
    """
    qa = _QA_SESSIONS.get(qa_token)
    if qa is None:
        raise HTTPException(404, "unknown or expired qa_token (a backend restart clears all Q&A sessions)")

    try:
        turn = ask(qa, body.question)
    except CoherenceError as e:
        # The turn cap was reached: a client error, not a server error
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"qa turn error: {e}")

    return {
        "answer": turn.answer_out.answer if turn.answer_out else None,
        "cited_fields": turn.answer_out.cited_fields if turn.answer_out else [],
        "answerable": turn.answer_out.answerable if turn.answer_out else False,
        "category": turn.category,
        "guardrail": turn.guardrail,
        "source": turn.source,
        "llm_calls": turn.llm_calls,
        "wall_s": turn.wall_s,
    }
