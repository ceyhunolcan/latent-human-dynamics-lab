"""Missingness features.

Missingness in passive sensing is not random. Sensor dropout often precedes
or follows behavioural change, so it carries signal in its own right. These
features encode it without imputation so downstream models can use the
pattern itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logging import get_logger

log = get_logger(__name__)

_FLAG_COLS = ("missing_wearable_flag", "missing_phone_flag", "missing_survey_flag")


def _consec_run(flag: pd.Series, pid: pd.Series) -> pd.Series:
    out = np.zeros(len(flag), dtype=float)
    run = 0
    prev_pid = None
    f = flag.fillna(0).astype(int).to_numpy()
    p = pid.to_numpy()
    for i in range(len(f)):
        if p[i] != prev_pid:
            run = 0
            prev_pid = p[i]
        run = run + 1 if f[i] == 1 else 0
        out[i] = run
    return pd.Series(out, index=flag.index)


def _rolling_mean(s: pd.Series, pid: pd.Series, window: int) -> pd.Series:
    return (
        s.groupby(pid, sort=False)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def compute_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add missingness pattern features.

    Adds:
      - total_missing_modalities (if not already present)
      - missingness_rate_7d, _14d
      - consecutive_missing_wearable_days
      - missingness_entropy (across the 3 modality flags, per row)
      - missingness_pressure (rolling instability of the missingness pattern)
    """
    if df.empty:
        return df

    out = df.copy()
    pid = out["participant_id"]

    # Ensure flags exist; default to 0 if not.
    for c in _FLAG_COLS:
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0).astype(int)

    if "total_missing_modalities" not in out.columns:
        out["total_missing_modalities"] = out[list(_FLAG_COLS)].sum(axis=1)

    out["missingness_rate_7d"] = _rolling_mean(
        out["total_missing_modalities"] / 3.0, pid, window=7
    )
    out["missingness_rate_14d"] = _rolling_mean(
        out["total_missing_modalities"] / 3.0, pid, window=14
    )

    out["consecutive_missing_wearable_days"] = _consec_run(out["missing_wearable_flag"], pid)

    # Per-row Shannon-style entropy across modality flags. With only 3 binary
    # flags, this is bounded above by log(3); we normalise to [0, 1].
    flags = out[list(_FLAG_COLS)].to_numpy()
    p_present = 1.0 - flags  # fraction present per modality
    eps = 1e-9
    p_norm = (p_present + flags + eps)  # always 1+eps, since each is 0 or 1
    # Compute entropy over the binomial "present vs missing" per modality, then
    # average across modalities. This yields a smooth scalar that is high when
    # the per-modality presence is uncertain across the row.
    per_mod_entropy = -(
        (p_present + eps) * np.log(p_present + eps)
        + (flags + eps) * np.log(flags + eps)
    )
    out["missingness_entropy"] = per_mod_entropy.mean(axis=1) / np.log(2)

    # Missingness pressure: rolling std of the total_missing_modalities counter.
    # Captures regime instability in sensor compliance.
    out["missingness_pressure"] = (
        out["total_missing_modalities"]
        .groupby(pid, sort=False)
        .rolling(window=7, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    log.debug("Missingness features added.")
    return out
