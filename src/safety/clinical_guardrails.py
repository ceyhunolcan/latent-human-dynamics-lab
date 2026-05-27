"""End-of-pipeline guardrails.

``validate_safe_output`` is the function every consumer (API, dashboard,
scripts) should call last. It:

1. sanitizes risk language,
2. attaches the disclaimer,
3. raises a ``SafetyError`` if reserved clinical language survives
   sanitization (this is a programmer error — it indicates that we
   constructed an output we shouldn't have).
"""

from __future__ import annotations

from typing import Any, Dict

from .output_disclaimer import attach_non_clinical_warning
from .risk_language import find_unsafe_terms, sanitize_clinical_language


class SafetyError(RuntimeError):
    """Raised when an output cannot be made safe by sanitization alone."""


def _sanitize_in_place(payload: Any) -> Any:
    """Recursively sanitize string values in dicts / lists."""
    if isinstance(payload, str):
        return sanitize_clinical_language(payload)
    if isinstance(payload, dict):
        return {k: _sanitize_in_place(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_in_place(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_sanitize_in_place(v) for v in payload)
    return payload


def _collect_strings(payload: Any) -> str:
    """Collect every string in a nested payload, joined by whitespace."""
    parts: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(payload)
    return " ".join(parts)


def validate_safe_output(
    payload: Dict[str, Any] | str,
    *,
    strict: bool = True,
) -> Dict[str, Any] | str:
    """Sanitize an output payload and attach the non-clinical warning.

    The scan for surviving reserved terms runs **before** the canonical
    disclaimer is attached, so the disclaimer's own use of the reserved
    vocabulary ("medical advice", "diagnosis", "treatment", "medical device")
    does not cause a false positive.

    Returns
    -------
    sanitized_payload : same type as input
        Output with clinical language rewritten and disclaimer attached.

    Raises
    ------
    SafetyError
        In ``strict=True`` mode (default), if reserved clinical patterns
        remain after sanitization.
    """
    sanitized = _sanitize_in_place(payload)

    flat = _collect_strings(sanitized)
    survivors = find_unsafe_terms(flat)
    if strict and survivors:
        raise SafetyError(
            f"Output contained reserved clinical terms after sanitization: {survivors}"
        )

    return attach_non_clinical_warning(sanitized)
