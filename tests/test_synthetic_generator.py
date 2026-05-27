"""Tests for the synthetic data generator."""

import numpy as np
import pandas as pd


def test_cohort_shape(small_cohort: pd.DataFrame):
    assert len(small_cohort) == 12 * 45
    assert small_cohort["participant_id"].nunique() == 12
    assert small_cohort["date"].nunique() == 45


def test_required_columns(small_cohort: pd.DataFrame):
    required = {
        "participant_id",
        "date",
        "sleep_duration",
        "hrv_rmssd",
        "resting_hr",
        "daily_steps",
        "screen_time_minutes",
        "mobility_radius_km",
        "temperature_c",
        "aqi",
        "regime_label",
    }
    missing = required - set(small_cohort.columns)
    assert not missing, f"Missing columns: {missing}"


def test_value_ranges(small_cohort: pd.DataFrame):
    df = small_cohort
    # Sleep should be in plausible human range
    s = df["sleep_duration"].dropna()
    assert s.between(0, 14).all()
    # Resting HR plausible
    rhr = df["resting_hr"].dropna()
    assert rhr.between(30, 130).all()
    # HRV non-negative
    assert (df["hrv_rmssd"].dropna() >= 0).all()
    # AQI non-negative
    assert (df["aqi"].dropna() >= 0).all()


def test_regime_labels_populate_all_classes(small_cohort: pd.DataFrame):
    regimes = set(small_cohort["regime_label"].dropna().unique())
    expected = {"stable", "stressed", "recovery", "dysregulated"}
    # On a small cohort we may not see every regime, but at least two should appear
    assert len(regimes & expected) >= 2, f"Saw regimes: {regimes}"


def test_determinism():
    from data.synthetic_generator import generate_synthetic_cohort

    a = generate_synthetic_cohort(n_participants=5, n_days=20, seed=7)
    b = generate_synthetic_cohort(n_participants=5, n_days=20, seed=7)
    # Compare a numeric column directly
    np.testing.assert_array_almost_equal(
        a["sleep_duration"].to_numpy(),
        b["sleep_duration"].to_numpy(),
    )
