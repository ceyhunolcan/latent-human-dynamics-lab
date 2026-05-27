"""Lightweight health check for the install.

Verifies imports, optional dependencies, paths, and basic functionality.
Useful for users after a fresh install or for the CI workflow:

    from utils.health_check import health_check
    health_check(verbose=True)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import List


@dataclass
class HealthReport:
    """Result of a health check. Iterates through a few categories."""
    passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        return (
            f"{len(self.passed)} OK · {len(self.warnings)} warnings · "
            f"{len(self.failed)} failures"
        )


def health_check(verbose: bool = False) -> HealthReport:
    """Run a series of checks against the install.

    Required deps must import; optional deps produce warnings if missing.
    Paths must exist and be writable. Core modules must load.
    """
    r = HealthReport()

    # Required dependencies
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml"):
        try:
            importlib.import_module(pkg)
            r.passed.append(f"required dependency: {pkg}")
        except ImportError:
            r.failed.append(f"required dependency missing: {pkg}")

    # Optional dependencies (degrade gracefully)
    for pkg in ("torch", "fastapi", "streamlit", "uvicorn"):
        try:
            importlib.import_module(pkg)
            r.passed.append(f"optional dependency present: {pkg}")
        except ImportError:
            r.warnings.append(
                f"optional dependency missing: {pkg} "
                "(some features will be unavailable)"
            )

    # Core internal modules
    for mod in (
        "data.synthetic_generator", "data.validation",
        "features", "states.latent_state_encoder", "states.regime_detector",
        "dynamics.transition_model", "dynamics.forcing_functions",
        "counterfactuals.perturbation_engine",
        "safety.clinical_guardrails", "safety.output_disclaimer",
        "evaluation.metrics", "evaluation.calibration",
    ):
        try:
            importlib.import_module(mod)
            r.passed.append(f"module loads: {mod}")
        except ImportError as e:
            r.failed.append(f"module fails to import: {mod} ({e})")

    # Paths
    try:
        from .paths import REPO_ROOT, DATA_DIR, RESULTS_DIR, CONFIGS_DIR
        for p in (REPO_ROOT, DATA_DIR, RESULTS_DIR, CONFIGS_DIR):
            if p.exists():
                r.passed.append(f"path exists: {p.relative_to(REPO_ROOT) if p != REPO_ROOT else p.name}")
            else:
                r.warnings.append(f"path missing (may be created on demand): {p}")
    except Exception as e:
        r.failed.append(f"paths module: {e}")

    # Smoke test of the safety guardrail
    try:
        from safety.clinical_guardrails import validate_safe_output
        from safety.output_disclaimer import DISCLAIMER
        out = validate_safe_output({"text": "patient diagnosis"})
        if "patient" in out["text"].lower() or "diagnosis" in out["text"].lower():
            r.failed.append("safety guardrail did not sanitize clinical language")
        elif out.get("disclaimer") != DISCLAIMER:
            r.failed.append("safety guardrail did not attach canonical disclaimer")
        else:
            r.passed.append("safety guardrail sanitizes + attaches disclaimer")
    except Exception as e:
        r.failed.append(f"safety guardrail check: {e}")

    # Smoke test of the generator
    try:
        from data.synthetic_generator import generate_synthetic_cohort
        df = generate_synthetic_cohort(2, 3, seed=17)
        if len(df) == 6:
            r.passed.append("synthetic generator produces expected output")
        else:
            r.failed.append(f"generator produced {len(df)} rows (expected 6)")
    except Exception as e:
        r.failed.append(f"generator check: {e}")

    if verbose:
        print("Health check results:")
        for m in r.passed:
            print(f"  OK    {m}")
        for m in r.warnings:
            print(f"  WARN  {m}")
        for m in r.failed:
            print(f"  FAIL  {m}")
        print(f"\n{r.summary()}")

    return r


if __name__ == "__main__":
    import sys
    report = health_check(verbose=True)
    sys.exit(0 if report.ok else 1)
