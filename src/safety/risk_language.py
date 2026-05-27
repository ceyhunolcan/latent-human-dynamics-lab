"""Mapping clinical-decision language to research-prototype language.

Clinical language is dangerous in two ways: it can mislead users about
what the model is doing, and it can imply medical authority the model
does not have. The replacements below are deliberately blunt — they
prefer false positives (over-sanitization) over false negatives.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable

# Mapping from (regex pattern, replacement). Order matters: more specific
# multi-word patterns are listed first.
RISK_LANGUAGE_REPLACEMENTS: Dict[str, str] = {
    r"\bmedical recommendation\b": "non-clinical simulation",
    r"\bdisease prediction\b": "proxy outcome estimate",
    r"\bdiagnos(?:is|e[sd]?)\b": "research signal",
    r"\btreatment[s]?\b": "perturbation scenario",
    r"\bprescription[s]?\b": "research scenario",
    r"\bprescribe[sd]?\b": "research-suggest",
    r"\bclinical advice\b": "research framing",
    r"\bmedical advice\b": "research framing",
    r"\bcure[sd]?\b": "improvement signal",
    r"\bprognos(?:is|tic)\b": "trajectory estimate",
    r"\bdosage\b": "magnitude",
    r"\bdose\b": "magnitude",
    r"\bpatient[s]?\b": "participant",
    r"\bsymptom[s]?\b": "observed signal",
    r"\bdoctor[s]?\b": "researcher",
    r"\bphysician[s]?\b": "researcher",
}


def sanitize_clinical_language(text: str) -> str:
    """Run every replacement on ``text`` and return the rewritten string."""
    if not text:
        return text
    out = text
    for pattern, replacement in RISK_LANGUAGE_REPLACEMENTS.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def convert_risk_language_to_research_language(
    text: str,
    extra_replacements: Dict[str, str] | None = None,
) -> str:
    """Alias of :func:`sanitize_clinical_language` with optional extras."""
    out = sanitize_clinical_language(text)
    if extra_replacements:
        for pattern, replacement in extra_replacements.items():
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def find_unsafe_terms(text: str, extra_terms: Iterable[str] = ()) -> list[str]:
    """Return a list of unsafe terms still present in ``text`` after
    sanitization. Used by :func:`safety.clinical_guardrails.validate_safe_output`.
    """
    if not text:
        return []
    reserved = list(RISK_LANGUAGE_REPLACEMENTS.keys()) + list(extra_terms)
    hits: list[str] = []
    for pattern in reserved:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits
