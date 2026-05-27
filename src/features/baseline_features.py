"""Personal baseline features.

Population-level normalisation throws away the most useful signal in passive
sensing data: individual variation. These functions express each daily reading
as a deviation from that person's own running baseline, so the resulting
features mean roughly the same thing whether the participant is a 19-year-old
endurance athlete or a 62-year-old with low resting activity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logging import get_logger

log = get_logger(__name__)

# Channels we compute a personal baseline for. Each gets a within-person
# rolling mean, a within-person z-score, and an anomaly score.
_PERSONAL_CHANNELS = (
    "sleep_duration",
    "sleep_efficiency",
    "hrv_rmssd",
    "resting_hr",
    "daily_steps",
    "active_minutes",
    "recovery_score",
    "stress_score",
    "screen_time_minutes",
    "phone_unlock_count",
    "mobility_radius_km",
    "behavioral_regularity",
)


def _personal_zscore(s: pd.Series, pid: pd.Series, window: int = 21) -> pd.Series:
    """Z-score against a person's own rolling mean and std."""
    grouped = s.groupby(pid, sort=False)
    mu = grouped.rolling(window=window, min_periods=5).mean().reset_index(level=0, drop=True)
    sd = grouped.rolling(window=window, min_periods=5).std().reset_index(level=0, drop=True)
    sd = sd.replace(0, np.nan)
    z = (s - mu) / sd
    return z.fillna(0.0)


def _personal_rolling_baseline(s: pd.Series, pid: pd.Series, window: int = 28) -> pd.Series:
    return (
        s.groupby(pid, sort=False)
        .rolling(window=window, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )


def compute_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute within-person baselines, deviations, and anomaly scores.

    For each channel in `_PERSONAL_CHANNELS` that is present, adds:
      - {channel}_personal_baseline_28d
      - {channel}_personal_z (21-day window)
      - {channel}_personal_anomaly  (clip(|z|, 0, 4))

    Also computes a compact per-row personalized_anomaly_score that averages
    the anomaly columns. This is intentionally bounded so an outlier on one
    channel cannot dominate the composite signal.
    """
    if df.empty:
        return df

    out = df.copy()
    pid = out["participant_id"]

    anomaly_cols = []
    for col in _PERSONAL_CHANNELS:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        baseline = _personal_rolling_baseline(s, pid)
        z = _personal_zscore(s, pid)
        anomaly = z.abs().clip(upper=4.0)

        out[f"{col}_personal_baseline_28d"] = baseline
        out[f"{col}_personal_z"] = z
        out[f"{col}_personal_anomaly"] = anomaly
        anomaly_cols.append(f"{col}_personal_anomaly")

    if anomaly_cols:
        out["personalized_anomaly_score"] = out[anomaly_cols].mean(axis=1)
    else:
        out["personalized_anomaly_score"] = 0.0

    log.debug("Personal baseline features added (%d channels).", len(anomaly_cols))
    return out
