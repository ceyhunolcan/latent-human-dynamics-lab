"""Clinical-safety guardrails for outputs of the latent-human-dynamics lab.

Every user-facing surface (API responses, dashboard panels, demo scripts,
generated reports) routes its output through these utilities before being
shown. The package implements three small but strict policies:

1. A non-clinical research disclaimer is attached to every output.
2. Clinical-decision language is rewritten into research language.
3. Any sentence containing reserved clinical terms (diagnosis, treatment,
   prescription, dose, prognosis) is flagged.

These are conservative defaults. They do not make the system safe for
clinical use; they prevent the prototype from being mistaken for one.
"""

from .output_disclaimer import (
    DISCLAIMER,
    attach_non_clinical_warning,
)
from .clinical_guardrails import validate_safe_output
from .risk_language import (
    RISK_LANGUAGE_REPLACEMENTS,
    sanitize_clinical_language,
    convert_risk_language_to_research_language,
)

__all__ = [
    "DISCLAIMER",
    "attach_non_clinical_warning",
    "validate_safe_output",
    "RISK_LANGUAGE_REPLACEMENTS",
    "sanitize_clinical_language",
    "convert_risk_language_to_research_language",
]
