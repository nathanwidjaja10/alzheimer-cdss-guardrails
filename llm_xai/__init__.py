"""llm_xai: the LLM-as-XAI layer (guardrailed narratives and Q&A for the CDSS).

The package only imports and wraps the code in `backend/` at the repository root and never
modifies it.

Structure:
    backend/   core modules (fact builder, guardrails, LLM client, FHIR wrapper, API router)
    tools/     evaluation scripts (gold facts, benchmarks, statistics)
    audit/     human-audit tooling (manifest builder, server, UI, summary)
    notebooks/ development drivers and paper figures
    configs/   model roster and Ollama Modelfile
    docs/      Q&A scope policy
    results/   evaluation outputs
"""

__version__ = "0.1.0"
