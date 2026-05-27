"""Climate and environmental features.

These features turn raw temperature, humidity, and AQI readings into a small
set of physiologically meaningful signals: cumulative heat burden, nighttime
heat stress, exposure duration during heat-wave episodes, and an overall
Environmental Physiological Load (EPL) score that combines them.

EPL is a research construct, not a clinical index. The weighting reflects the
literature on the relative impact of daytime heat, nighttime heat, air quality,
and heat-wave events on autonomic recovery and sleep, but no claim is made
about prognostic value for any individual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logging import get_logger

log = get_logger(__name__)


def _rolling_by_pid(s: pd.Series, pid: pd.Series, window: int, min_periods: int = 1):
    return s.groupby(pid, sort=False).rolling(window=window, min_periods=min_periods).mean().reset_index(level=0, drop=True)


def _rolling_sum_by_pid(s: pd.Series, pid: pd.Series, window: int):
    return s.groupby(pid, sort=False).rolling(window=window, min_periods=1).sum().reset_index(level=0, drop=True)


def _consecutive_run_length(flag: pd.Series, pid: pd.Series) -> pd.Series:
    """For each row, return the length of the current consecutive run of 1s
    in flag within that participant. Resets to 0 on a 0.
    """
    out = np.zeros(len(flag), dtype=float)
    run = 0
    prev_pid = None
    flag_vals = flag.fillna(0).astype(int).to_numpy()
    pid_vals = pid.to_numpy()
    for i in range(len(flag_vals)):
        if pid_vals[i] != prev_pid:
            run = 0
            prev_pid = pid_vals[i]
        run = run + 1 if flag_vals[i] == 1 else 0
        out[i] = run
    return pd.Series(out, index=flag.index)


def compute_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add climate-derived features to a per-participant, per-day dataframe.

    Expects columns: participant_id, date, temperature_c, nighttime_temperature_c,
    humidity, aqi, heat_index, heat_wave_flag.

    Adds:
      - heat_index_recompute (sanity column when heat_index is missing)
      - cumulative_heat_burden_7d, _14d (rolling sum of heat_index above 28°C)
      - nighttime_heat_stress_7d
      - aqi_burden_7d, _14d
      - heat_wave_exposure_days (current consecutive run length)
      - environmental_physiological_load (EPL composite, z-score scaled)
    """
    if df.empty:
        return df

    out = df.copy()
    pid = out["participant_id"]

    # Recompute heat index if absent. Simple humidex-style proxy.
    if "heat_index" not in out.columns or out["heat_index"].isna().all():
        humidity = out.get("humidity", pd.Series(50.0, index=out.index)).fillna(50.0)
        temp = out["temperature_c"].fillna(out["temperature_c"].median())
        out["heat_index"] = temp + 0.07 * (humidity - 50.0).clip(lower=0)

    # Heat exposure above a 28°C comfort threshold. We do not pretend this is a
    # clinical threshold; it is a working baseline used widely in environmental
    # physiology literature for tropical/temperate comfort studies.
    heat_excess = (out["heat_index"] - 28.0).clip(lower=0)
    out["cumulative_heat_burden_7d"] = _rolling_sum_by_pid(heat_excess, pid, window=7)
    out["cumulative_heat_burden_14d"] = _rolling_sum_by_pid(heat_excess, pid, window=14)

    if "nighttime_temperature_c" in out.columns:
        night_excess = (out["nighttime_temperature_c"] - 22.0).clip(lower=0)
        out["nighttime_heat_stress_7d"] = _rolling_sum_by_pid(night_excess, pid, window=7)
    else:
        out["nighttime_heat_stress_7d"] = 0.0

    if "aqi" in out.columns:
        aqi_excess = (out["aqi"] - 75.0).clip(lower=0)
        out["aqi_burden_7d"] = _rolling_sum_by_pid(aqi_excess, pid, window=7)
        out["aqi_burden_14d"] = _rolling_sum_by_pid(aqi_excess, pid, window=14)
    else:
        out["aqi_burden_7d"] = 0.0
        out["aqi_burden_14d"] = 0.0

    if "heat_wave_flag" in out.columns:
        out["heat_wave_exposure_days"] = _consecutive_run_length(out["heat_wave_flag"], pid)
    else:
        out["heat_wave_exposure_days"] = 0.0

    # Environmental Physiological Load. We z-score each component over the
    # cohort, weight per the configured priors, and sum. This is intentionally
    # auditable so reviewers can interrogate every coefficient.
    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sd

    z_day = _z(out["cumulative_heat_burden_7d"])
    z_night = _z(out["nighttime_heat_stress_7d"])
    z_aqi = _z(out["aqi_burden_7d"])
    z_hw = _z(out["heat_wave_exposure_days"])

    out["environmental_physiological_load"] = (
        0.30 * z_day + 0.30 * z_night + 0.25 * z_aqi + 0.15 * z_hw
    )

    log.debug("Climate features added.")
    return out
