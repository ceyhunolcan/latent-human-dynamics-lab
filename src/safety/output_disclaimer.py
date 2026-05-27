"""The single canonical non-clinical disclaimer."""

from __future__ import annotations

from typing import Any, Dict

DISCLAIMER: str = (
    "Research prototype only. Not medical advice, diagnosis, treatment, "
    "or a medical device."
)


def attach_non_clinical_warning(payload: Dict[str, Any] | str) -> Dict[str, Any] | str:
    """Attach the disclaimer to a payload.

    - If ``payload`` is a dict, set/overwrite ``payload['disclaimer']``.
    - If ``payload`` is a string, append the disclaimer on a new line.
    - Otherwise, return as-is (callers should typically wrap in a dict).
    """
    if isinstance(payload, dict):
        out = dict(payload)
        out["disclaimer"] = DISCLAIMER
        return out
    if isinstance(payload, str):
        if DISCLAIMER in payload:
            return payload
        return f"{payload}\n\n{DISCLAIMER}"
    return payload
