"""Shared pytest fixtures and sys.path setup for the test suite."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def small_cohort() -> pd.DataFrame:
    """A small but realistic synthetic cohort, computed once per test session."""
    from data.synthetic_generator import generate_synthetic_cohort

    return generate_synthetic_cohort(n_participants=12, n_days=45, seed=17)


@pytest.fixture(scope="session")
def engineered_cohort(small_cohort: pd.DataFrame) -> pd.DataFrame:
    from features import engineer_all_features

    return engineer_all_features(small_cohort)


@pytest.fixture(scope="session")
def latent_states(engineered_cohort: pd.DataFrame) -> np.ndarray:
    from states.latent_state_encoder import encode_latent_states_classical

    df = engineered_cohort

    def pick(cols):
        present = [c for c in cols if c in df.columns]
        return (
            df[present].to_numpy(dtype=float, na_value=0.0)
            if present
            else np.zeros((len(df), len(cols)))
        )

    W = pick(["sleep_duration_hours", "hrv_rmssd", "resting_hr", "daily_steps", "recovery_score"])
    B = pick(["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"])
    C = pick(["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"])
    M = pick(["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"])
    P = pick(["baseline_hrv", "baseline_resting_hr", "baseline_climate_vulnerability", "baseline_resilience"])
    return encode_latent_states_classical(W, B, C, M, P).latent
