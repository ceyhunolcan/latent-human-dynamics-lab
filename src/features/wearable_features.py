"""Wearable-derived features.

Each participant's HRV, resting HR, sleep, and recovery streams are
combined with their personal baseline to produce within-person deviations
and rolling trends. The features are intended as inputs to the latent
state encoder, not as standalone biomarkers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_std(s: pd.Series) -> float:
    return float(s.std()) if s.notna().sum() > 1 else 1.0


def compute_wearable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add wearable-derived features to ``df`` (per-participant)."""
    out = df.copy().sort_values(["participant_id", "date"]).reset_index(drop=True)
    grp = out.groupby("participant_id", group_keys=False)

    # Deviation from baseline
    if "hrv_rmssd" in out and "baseline_hrv" in out:
        out["hrv_deviation"] = (out["hrv_rmssd"] - out["baseline_hrv"]) / out["baseline_hrv"].replace(0, np.nan)
    if "resting_hr" in out and "baseline_resting_hr" in out:
        out["rhr_deviation"] = (out["resting_hr"] - out["baseline_resting_hr"]) / out["baseline_resting_hr"].replace(0, np.nan)

    # Rolling sleep statistics
    if "sleep_duration" in out:
        out["sleep_rolling_mean_7d"] = grp["sleep_duration"].transform(
            lambda s: s.rolling(7, min_periods=2).mean()
        )
        out["sleep_regularity_index"] = grp["sleep_midpoint"].transform(
            lambda s: -s.rolling(7, min_periods=3).std()
        ).fillna(0.0) if "sleep_midpoint" in out else 0.0

    # Stress burden = exponential moving average of stress score
    if "stress_score" in out:
        out["stress_burden_ewm"] = grp["stress_score"].transform(
            lambda s: s.ewm(halflife=3, adjust=False).mean()
        )

    # Trends as 7-day slope (cheap proxy: rolling mean diff)
    for col in ("recovery_score", "active_minutes"):
        if col in out:
            roll = grp[col].transform(lambda s: s.rolling(7, min_periods=3).mean())
            out[f"{col}_trend_7d"] = roll - roll.shift(7).where(
                out["participant_id"] == out["participant_id"].shift(7)
            )

    return out
