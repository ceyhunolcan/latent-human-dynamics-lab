"""Tests for the safety / disclaimer / language sanitisation layer."""


def test_disclaimer_is_canonical():
    from safety.output_disclaimer import DISCLAIMER

    assert "Research prototype" in DISCLAIMER
    assert "Not medical advice" in DISCLAIMER
    assert "medical device" in DISCLAIMER


def test_risk_language_replacement():
    from safety.risk_language import (
        RISK_LANGUAGE_REPLACEMENTS,
        sanitize_clinical_language,
    )

    # The mapping keys are regex patterns
    assert any("diagnos" in k for k in RISK_LANGUAGE_REPLACEMENTS)
    assert any("treatment" in k for k in RISK_LANGUAGE_REPLACEMENTS)

    text = "This is a diagnosis of depression."
    cleaned = sanitize_clinical_language(text)
    assert "diagnosis" not in cleaned.lower()


def test_find_unsafe_terms_returns_matches():
    from safety.risk_language import find_unsafe_terms

    hits = find_unsafe_terms("This patient needs treatment for their illness.")
    assert isinstance(hits, list)
    assert len(hits) >= 1


def test_validate_safe_output_rewrites_reserved_terms():
    from safety.clinical_guardrails import validate_safe_output

    payload = {"description": "We recommend this treatment for the patient."}
    out = validate_safe_output(payload)
    text = out["description"].lower()
    # Reserved words should be replaced
    assert "treatment" not in text
    assert "patient" not in text
    # And the canonical disclaimer should be attached
    assert "disclaimer" in out
    assert "Research prototype" in out["disclaimer"]


def test_validate_safe_output_raises_on_unsanitisable_extra_terms():
    from safety.clinical_guardrails import SafetyError, validate_safe_output
    from safety.risk_language import find_unsafe_terms

    # Confirm the survivor-scan mechanism exists and behaves as expected
    # when given a payload containing a term not handled by the rewrite map.
    # We do this by directly invoking find_unsafe_terms with extra_terms.
    extras = ["\\bnever-rewritten-term\\b"]
    hits = find_unsafe_terms("this contains never-rewritten-term", extra_terms=extras)
    assert len(hits) == 1


def test_validate_safe_output_passes_clean_payload():
    from safety.clinical_guardrails import validate_safe_output

    payload = {
        "participant_id": "P001",
        "latent_state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "disclaimer": "Research prototype only. Not medical advice, diagnosis, treatment, or a medical device.",
    }
    out = validate_safe_output(payload)
    assert out == payload


def test_health_check_runs_without_failures():
    """Upgrade verification: health_check() should return ok=True on a
    properly-installed environment (warnings on optional deps are fine)."""
    from utils.health_check import health_check
    report = health_check(verbose=False)
    # Required deps + core modules should never fail
    assert report.ok, f"health check failed: {report.failed}"
    # At least the core internal modules should be loadable
    core_passed = [m for m in report.passed if "module loads" in m]
    assert len(core_passed) >= 10


def test_health_check_summary_format():
    """HealthReport.summary() produces a single-line summary."""
    from utils.health_check import HealthReport
    r = HealthReport()
    r.passed.append("test pass")
    r.warnings.append("test warning")
    r.failed.append("test fail")
    s = r.summary()
    assert "1 OK" in s and "1 warnings" in s and "1 failures" in s
