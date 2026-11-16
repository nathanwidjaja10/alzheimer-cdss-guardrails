"""Three-tier neuroanatomy whitelist used by the guardrail to reject fabricated brain structures.

- Tier A: structures measured by the source model; may be named and given a number.
- Tier B: allowed anatomical context terms (e.g. "medial temporal lobe"); may be named but
  not given a number.
- Tier C: all other structures (putamen, thalamus, entorhinal cortex, ...); naming one is a
  `fabricated_structure` violation.

`_FS_ID_TO_NAME` mirrors `backend.inference_engine.ROI_LABELS` and is hardcoded so that this
module does not import the model; keep both in sync.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Tier A: FreeSurfer label id -> structure name
# ---------------------------------------------------------------------------
_FS_ID_TO_NAME = {
    17: "hippocampus", 53: "hippocampus",
    4: "lateral ventricle", 43: "lateral ventricle",
    5: "inferior lateral ventricle", 44: "inferior lateral ventricle",
    14: "third ventricle",
    18: "amygdala", 54: "amygdala",
    3: "cerebral cortex", 42: "cerebral cortex",
}


def _tier_a_terms() -> set[str]:
    """Tier-A vocabulary: measured structures plus slice-plane names."""
    names = set(_FS_ID_TO_NAME.values())
    # Plane names may carry a number (per-plane attention share)
    names |= {"coronal", "axial", "sagittal",
              "coronal plane", "axial plane", "sagittal plane"}
    return names


TIER_A: set[str] = _tier_a_terms()

TIER_B: set[str] = {
    "medial temporal lobe", "temporal lobe", "temporal horn",
    "cerebral atrophy", "cortical atrophy", "global cortical atrophy",
    "ventriculomegaly", "ventricular enlargement", "ventricular system",
    "gray matter", "grey matter", "white matter",
    "sulcal widening", "sulci", "gyri",
}

TIER_C: set[str] = {
    "putamen", "caudate", "caudate nucleus", "globus pallidus", "lentiform nucleus",
    "thalamus", "hypothalamus", "pons", "midbrain", "medulla", "brainstem",
    "cerebellum", "cerebellar vermis", "corpus callosum", "insula", "insular cortex",
    "cingulate", "cingulate gyrus", "posterior cingulate",
    "entorhinal cortex", "perirhinal cortex", "parahippocampal gyrus",
    "precuneus", "fornix", "mammillary body", "nucleus accumbens",
    "substantia nigra", "internal capsule", "external capsule", "claustrum",
    "frontal lobe", "parietal lobe", "occipital lobe",
    "basal ganglia", "optic chiasm", "pineal gland", "pituitary",
    # Desikan-Killiany atlas sub-regions: plausible-sounding structures the model does not measure
    "inferior parietal lobule", "superior parietal lobule", "paracentral lobule",
    "superior temporal gyrus", "middle temporal gyrus", "inferior temporal gyrus",
    "transverse temporal gyrus", "temporal pole",
    "superior frontal gyrus", "middle frontal gyrus", "inferior frontal gyrus",
    "orbitofrontal cortex", "frontal pole",
    "precentral gyrus", "postcentral gyrus", "supramarginal gyrus", "angular gyrus",
    "fusiform gyrus", "lingual gyrus", "cuneus",
    "pars triangularis", "pars opercularis", "pars orbitalis",
}

# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _norm(term: str) -> str:
    """Lower-case a term and collapse whitespace."""
    return _WS.sub(" ", term.strip().lower())


def _compile(terms: set[str]) -> re.Pattern:
    """Compile a whole-word alternation of the terms, longest first."""
    ordered = sorted({_norm(t) for t in terms}, key=len, reverse=True)
    if not ordered:
        return re.compile(r"(?!x)x")  # never matches
    body = "|".join(re.escape(t) for t in ordered)
    return re.compile(rf"\b(?:{body})\b", re.IGNORECASE)


_RX: dict[str, re.Pattern] = {"A": _compile(TIER_A), "B": _compile(TIER_B), "C": _compile(TIER_C)}
_RX_FULL: dict[str, re.Pattern] = {
    k: re.compile(rf"^(?:{v.pattern})$", re.IGNORECASE) for k, v in _RX.items()
}


def classify_term(term: str) -> str | None:
    """Return 'A', 'B', 'C' or None for a single term.

    Tier C is checked first so that a specific phrase ('entorhinal cortex') is not classified
    by a Tier-A token it contains.
    """
    t = _norm(term)
    for tier in ("C", "A", "B"):
        if _RX_FULL[tier].match(t):
            return tier
    return None


def scan(text: str) -> list[dict]:
    """Return non-overlapping {term, tier, start, end} matches found in the text.

    Tiers are matched in the order C, B, A so the most specific span claims a position first.
    """
    hits: list[dict] = []
    claimed: list[tuple[int, int]] = []
    for tier in ("C", "B", "A"):
        for m in _RX[tier].finditer(text):
            s, e = m.start(), m.end()
            if any(not (e <= cs or s >= ce) for cs, ce in claimed):
                continue
            claimed.append((s, e))
            hits.append({"term": m.group(0), "tier": tier, "start": s, "end": e})
    hits.sort(key=lambda h: h["start"])
    return hits
