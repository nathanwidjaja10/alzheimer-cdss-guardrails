"""llm_xai.backend: core modules of the LLM-XAI layer.

Importing this package adds the repository root to `sys.path`, so
`from backend.inference_engine import ...` resolves from notebooks, tools and uvicorn.
"""
from __future__ import annotations

import sys
from pathlib import Path

# .../llm_xai/backend/__init__.py -> parents[2] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Artifact versions, recorded in every report for reproducibility
SCHEMA_VERSION = "1.2.0"        # Fact JSON schema version
GUARDRAIL_VERSION = "1.3.0"     # rule version of the narrative guardrail (run_dag)
#   1.1.0  English correction messages and violation details.
#   1.2.0  APOE self-diagnosis pattern matched across sentence boundaries; anatomy tier C
#          extended with Desikan-Killiany sub-region terms.
#   1.3.0  Fact field `clinical_covariates.apoe_genotype` (general E2/E3/E4 genotype);
#          `dominant_structures` derived from saliency energy; `apoe_negation_contradiction`
#          pattern in audit_contradiction; `audit_structure_binding` checks slice/structure
#          pairs in narratives.

RESULTS_DIR = REPO_ROOT / "llm_xai" / "results"
CONFIGS_DIR = REPO_ROOT / "llm_xai" / "configs"

# Rule version of the Q&A guardrail (run_qa_dag), versioned separately so that Q&A changes
# do not invalidate the narrative cache, whose key uses GUARDRAIL_VERSION.
QA_GUARDRAIL_VERSION = "0.6.0"
#   0.1.0  audit_numeric, audit_anatomy, audit_contradiction, audit_grounding (local NLI
#          entailment against the cited fields), audit_scope (forces <REFUSE> outside scope),
#          coherence guard (immutable differential, turn cap).
#   0.2.0  audit_label_binding: label-to-number correspondence in dictionary fields.
#   0.3.0  audit_comparison: arithmetic check of "X greater/smaller than Y" claims.
#   0.4.0  A failing NLI grounding verdict is overridden when audit_comparison actively
#          verifies the claim as correct.
#   0.5.0  audit_superlative: argmax/argmin claims; wider comparison patterns.
#   0.6.0  audit_structure_binding added to the Q&A audit list.

__all__ = [
    "REPO_ROOT", "SCHEMA_VERSION", "GUARDRAIL_VERSION", "QA_GUARDRAIL_VERSION",
    "RESULTS_DIR", "CONFIGS_DIR",
]
