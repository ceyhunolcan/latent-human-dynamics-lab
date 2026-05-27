"""Behavioral features derived from screen, phone, and mobility streams."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["participant_id", "date"]).reset_index(drop=True)
    grp = out.groupby("participant_id", group_keys=False)

    if "screen_time_minutes" in out and "baseline_screen_time" in out:
        out["screen_deviation"] = (
            out["screen_time_minutes"] - out["baseline_screen_time"]
        ) / out["baseline_screen_time"].replace(0, np.nan)

    if "phone_unlock_count" in out:
        rolling_mean = grp["phone_unlock_count"].transform(
            lambda s: s.rolling(14, min_periods=4).mean()
        )
        out["phone_unlock_deviation"] = (out["phone_unlock_count"] - rolling_mean) / rolling_mean.replace(0, np.nan)

    if "mobility_radius_km" in out:
        out["mobility_entropy_7d"] = grp["mobility_radius_km"].transform(
            lambda s: s.rolling(7, min_periods=3).std()
        )

    if "behavioral_regularity" in out:
        out["behavioral_regularity_deviation"] = grp["behavioral_regularity"].transform(
            lambda s: s - s.rolling(14, min_periods=4).mean()
        )

    # Social rhythm instability: weekday-vs-weekend midpoint gap (rolling)
    if "sleep_midpoint" in out and "date" in out:
        out["_weekday"] = pd.to_datetime(out["date"]).dt.dayofweek
        out["_is_weekend"] = (out["_weekday"] >= 5).astype(int)
        weekend_mid = grp.apply(
            lambda g: g.loc[g["_is_weekend"] == 1, "sleep_midpoint"].rolling(2, min_periods=1).mean()
        )
        # Robust assignment: compute per-participant social-rhythm instability
        out["social_rhythm_instability"] = grp["sleep_midpoint"].transform(
            lambda s: s.rolling(14, min_periods=4).std()
        )
        out = out.drop(columns=["_weekday", "_is_weekend"])

    return out
