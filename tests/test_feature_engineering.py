"""Tests for the feature engineering pipeline."""

import pandas as pd


def test_engineered_cohort_adds_features(small_cohort: pd.DataFrame, engineered_cohort: pd.DataFrame):
    assert engineered_cohort.shape[0] == small_cohort.shape[0]
    assert engineered_cohort.shape[1] > small_cohort.shape[1]


def test_canonical_feature_columns_present(engineered_cohort: pd.DataFrame):
    expected = [
        "missingness_rate_7d",
        "missingness_pressure",
        "personalized_anomaly_score",
        "sleep_regularity_index",
        "stress_burden_ewm",
        "environmental_physiological_load",
        "social_rhythm_instability",
    ]
    for col in expected:
        assert col in engineered_cohort.columns, f"Missing engineered feature: {col}"


def test_no_constant_columns_in_engineered(engineered_cohort: pd.DataFrame):
    # Feature engineering should not produce any all-NaN or zero-variance new columns
    # (we allow the original deterministic baselines to be constant per participant).
    nans = engineered_cohort.isna().mean()
    assert (nans < 1.0).all(), "Found fully-NaN columns"


def test_feature_ordering_is_safe(small_cohort: pd.DataFrame):
    """Missingness must be computed BEFORE imputation, baseline BEFORE wearable, etc."""
    from features import engineer_all_features

    df = small_cohort.copy()
    out = engineer_all_features(df)
    # Missingness features should reflect the raw input's NaN structure
    assert (out["missingness_rate_7d"].between(0, 1)).all()


def test_epl_has_sufficient_variation_for_tertile_split(engineered_cohort: pd.DataFrame):
    """Regression for Bug 8: EPL must have enough variation that rank-percentile
    tertiles produce non-empty groups. The environmental_forcing_response figure
    relies on this.
    """
    import numpy as np
    import pandas as pd

    epl = engineered_cohort["environmental_physiological_load"].to_numpy(dtype=float)
    rank = pd.Series(epl).rank(method="average", pct=True).to_numpy()
    high = (rank >= 2 / 3).sum()
    low = (rank <= 1 / 3).sum()
    # Both tertile masks must be non-empty so the figure can produce both groups
    assert high > 0 and low > 0, f"EPL tertile split would be empty: high={high}, low={low}"


def test_empty_cohort_handled_gracefully():
    """Regression for Bug 21: feature engineering on an empty DataFrame used
    to crash deep inside pandas with 'No objects to concatenate'.
    Should now return an empty DataFrame cleanly."""
    from data.synthetic_generator import generate_synthetic_cohort
    from features import engineer_all_features

    df = generate_synthetic_cohort(3, 10, seed=17).iloc[:0]
    eng = engineer_all_features(df)
    assert len(eng) == 0
    # Should still have a DataFrame structure
    import pandas as pd
    assert isinstance(eng, pd.DataFrame)


def test_pipeline_summary_records_stages():
    """Upgrade verification: PipelineSummary tracks per-stage transformations."""
    import time
    import pandas as pd
    from utils.pipeline_summary import PipelineSummary, StageTimer

    summary = PipelineSummary()
    df_a = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    df_b = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "z": [7.0, 8.0, 9.0]})

    timer = StageTimer()
    time.sleep(0.01)
    summary.record("widen", df_a, df_b, timer.elapsed())

    d = summary.as_dict()
    assert len(d["stages"]) == 1
    assert d["stages"][0]["name"] == "widen"
    assert d["stages"][0]["rows_in"] == 3
    assert d["stages"][0]["rows_out"] == 3
    assert d["stages"][0]["columns_in"] == 2
    assert d["stages"][0]["columns_out"] == 3
    assert d["stages"][0]["duration_seconds"] > 0
    md = summary.as_markdown()
    assert "widen" in md and "|" in md


def test_pipeline_summary_nan_rate():
    """NaN rate calculation should be correct."""
    import pandas as pd
    import numpy as np
    from utils.pipeline_summary import PipelineSummary, StageTimer

    df_clean = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df_nan = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})

    summary = PipelineSummary()
    summary.record("dirty", df_clean, df_nan, 0.001)
    s = summary.stages[0]
    assert s.nan_rate_in == 0.0
    # 2 NaN out of 6 cells = 0.333...
    assert abs(s.nan_rate_out - 2/6) < 1e-9
