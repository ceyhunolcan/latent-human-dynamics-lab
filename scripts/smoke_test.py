"""End-to-end smoke test.

Runs the entire pipeline on a tiny cohort and verifies every artifact
landed where expected. Useful for CI and for verifying a fresh install.

Run from repo root::

    python scripts/smoke_test.py

Exits 0 on success, 1 on any failure. Prints a step-by-step summary.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from data.synthetic_generator import generate_synthetic_cohort
from data.validation import validate_cohort
from features import engineer_all_features
from states.latent_state_encoder import encode_latent_states_classical, LATENT_DIM_NAMES
from states.regime_detector import fit_regime_detector
from states.energy_landscape import estimate_energy_landscape
from states.early_warning import critical_transition_warning_score
from dynamics.transition_model import LatentDynamicsModel
from counterfactuals.perturbation_engine import (
    PerturbationSpec, simulate_perturbation, available_perturbations,
)
from safety.clinical_guardrails import validate_safe_output
from safety.output_disclaimer import DISCLAIMER


GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}    {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def main() -> int:
    print("Smoke test starting...")
    print()

    n_failed = 0

    # Stage 1: Generate
    try:
        df = generate_synthetic_cohort(n_participants=5, n_days=20, seed=17)
        assert len(df) == 100
        assert df["participant_id"].nunique() == 5
        _ok(f"Generator: 5×20 cohort, {len(df)} rows × {df.shape[1]} cols")
    except Exception as e:
        _fail(f"Generator: {e}")
        n_failed += 1
        return 1  # short-circuit; nothing downstream will work

    # Stage 2: Validate
    try:
        validate_cohort(df, raise_on_error=True)
        _ok("Validator: schema and ranges pass")
    except Exception as e:
        _fail(f"Validator: {e}")
        n_failed += 1

    # Stage 3: Feature engineering
    try:
        eng = engineer_all_features(df)
        assert len(eng) == len(df)
        assert eng.shape[1] > df.shape[1]
        _ok(f"Features: {df.shape[1]} → {eng.shape[1]} columns")
    except Exception as e:
        _fail(f"Features: {e}")
        n_failed += 1
        return 1

    # Stage 4: Encode
    try:
        def pk(c):
            p = [x for x in c if x in eng.columns]
            return eng[p].to_numpy(dtype=float, na_value=0.0)

        Z = encode_latent_states_classical(
            pk(["sleep_duration_hours", "hrv_rmssd", "resting_hr", "daily_steps", "recovery_score"]),
            pk(["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"]),
            pk(["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"]),
            pk(["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"]),
            pk(["baseline_hrv", "baseline_resting_hr", "baseline_climate_vulnerability", "baseline_resilience"]),
        ).latent
        assert Z.shape == (len(eng), 6)
        assert np.isfinite(Z).all()
        _ok(f"Encoder: {Z.shape} latent states, all finite")
    except Exception as e:
        _fail(f"Encoder: {e}")
        n_failed += 1
        return 1

    # Stage 5: Regime detection
    try:
        det = fit_regime_detector(Z, n_clusters=4, random_state=17)
        labels = det.predict(Z)
        n_regimes = len(set(labels))
        assert n_regimes >= 1
        _ok(f"Regime detector: {n_regimes} distinct regime(s) inferred")
    except Exception as e:
        _fail(f"Regime detector: {e}")
        n_failed += 1

    # Stage 6: Dynamics
    try:
        dyn = LatentDynamicsModel.from_config()
        z0 = Z[0]
        env = np.tile([20.0, 14.0, 50.0, 0.0], (14, 1))
        beh = np.zeros((14, 4))
        traj = dyn.simulate_trajectory(z0, env, beh)
        assert traj.shape == (15, 6)
        assert np.abs(traj).max() < 10
        _ok(f"Dynamics: 14-step trajectory bounded to |z|<{np.abs(traj).max():.2f}")
    except Exception as e:
        _fail(f"Dynamics: {e}")
        n_failed += 1

    # Stage 7: All 7 perturbations
    try:
        for ptype in available_perturbations():
            spec = PerturbationSpec(perturbation_type=ptype, horizon_days=14)
            res = simulate_perturbation(
                np.zeros(6), np.zeros((14, 4)), np.zeros((14, 4)),
                spec, dynamics_model=LatentDynamicsModel(rng_seed=17),
            )
            assert "disclaimer" in res.to_dict()
            assert np.isfinite(np.array(res.counterfactual_trajectory)).all()
        _ok(f"Counterfactuals: all 7 perturbations run cleanly")
    except Exception as e:
        _fail(f"Counterfactuals: {e}")
        n_failed += 1

    # Stage 8: Energy landscape
    try:
        L = estimate_energy_landscape(Z, grid_size=20)
        assert L.energy.shape == (20, 20)
        assert np.isfinite(L.energy).all()
        _ok(f"Energy landscape: 20×20 grid, all finite")
    except Exception as e:
        _fail(f"Energy landscape: {e}")
        n_failed += 1

    # Stage 9: Early warning
    try:
        # Need ≥ 14 rows for default window
        w = critical_transition_warning_score(Z[:14] if len(Z) >= 14 else Z)
        assert "warning_score" in w
        _ok("Early warning: warning_score computed")
    except Exception as e:
        _fail(f"Early warning: {e}")
        n_failed += 1

    # Stage 10: Safety guardrail
    try:
        out = validate_safe_output({"text": "patient diagnosis treatment"})
        assert "patient" not in out["text"].lower()
        assert "disclaimer" in out
        assert out["disclaimer"] == DISCLAIMER
        _ok("Safety: clinical language rewritten + disclaimer attached")
    except Exception as e:
        _fail(f"Safety: {e}")
        n_failed += 1

    print()
    if n_failed == 0:
        print(f"{GREEN}All 10 stages passed.{RESET}")
        print("(research prototype, non-clinical)")
        return 0
    else:
        print(f"{RED}{n_failed} stage(s) failed.{RESET}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
