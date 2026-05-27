"""Preprocessing: load, clean, sort, impute, split, save.

The preprocessing pipeline is intentionally simple and deterministic. The
heavy lifting (feature engineering, latent state inference, dynamics) lives
downstream. Here we just make sure inputs are well-formed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from utils.paths import (
    PROCESSED_DIR,
    SYNTHETIC_DIR,
    cohort_csv,
    ensure_dir,
)
from .synthetic_generator import generate_synthetic_cohort, save_synthetic_data
from .validation import validate_cohort


def load_raw_or_synthetic_data(
    path: str | Path | None = None,
    *,
    generate_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the cohort CSV; generate the default synthetic cohort if missing.

    Parameters
    ----------
    path : optional
        Path to a cohort CSV. If None, uses the default synthetic location.
    generate_if_missing : bool
        If True (default) and the file does not exist, generate the default
        synthetic cohort and write it to the default path.
    """
    target = Path(path) if path else cohort_csv()
    if not target.exists():
        if not generate_if_missing:
            raise FileNotFoundError(target)
        ensure_dir(SYNTHETIC_DIR)
        df = generate_synthetic_cohort()
        save_synthetic_data(df, path=target)
        return df
    df = pd.read_csv(target, parse_dates=["date"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop fully-empty rows, parse dates."""
    out = df.copy()
    if "date" in out.columns and not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    # Drop rows with no participant_id or date
    out = out.dropna(subset=["participant_id", "date"])
    return out.reset_index(drop=True)


def sort_by_participant_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["participant_id", "date"]).reset_index(drop=True)


def impute_safe_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative imputation only for downstream-fragile columns.

    Strategy: per-participant forward-fill on physiological columns
    capped at 3 days, then median imputation as a last resort. Categorical
    columns are filled with explicit "unknown" labels. Missingness flags
    are preserved exactly.
    """
    out = df.copy()
    physio = [
        "hrv_rmssd",
        "resting_hr",
        "sleep_duration",
        "sleep_efficiency",
        "sleep_midpoint",
        "daily_steps",
        "active_minutes",
        "recovery_score",
        "stress_score",
        "screen_time_minutes",
        "phone_unlock_count",
        "mobility_radius_km",
        "location_entropy",
        "mood_score",
        "fatigue_score",
        "perceived_stress",
        "energy_score",
        "cognitive_load_proxy",
    ]
    physio = [c for c in physio if c in out.columns]

    # per-participant forward fill up to 3 consecutive missing days
    out[physio] = (
        out.groupby("participant_id", group_keys=False)[physio]
        .apply(lambda g: g.ffill(limit=3))
    )
    # fall back to median per-participant
    out[physio] = (
        out.groupby("participant_id", group_keys=False)[physio]
        .apply(lambda g: g.fillna(g.median()))
    )
    # global median as final fallback
    for c in physio:
        out[c] = out[c].fillna(out[c].median())

    if "chronotype" in out.columns:
        out["chronotype"] = out["chronotype"].fillna("intermediate")
    if "sex" in out.columns:
        out["sex"] = out["sex"].fillna("NB")

    return out


def create_train_validation_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 17,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split *by participant* so that no participant appears in both sets."""
    rng = np.random.default_rng(seed)
    pids = df["participant_id"].drop_duplicates().to_numpy()
    rng.shuffle(pids)
    n_val = max(1, int(round(len(pids) * val_frac)))
    val_ids = set(pids[:n_val])
    train = df[~df["participant_id"].isin(val_ids)].reset_index(drop=True)
    val = df[df["participant_id"].isin(val_ids)].reset_index(drop=True)
    return train, val


def save_processed_data(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write processed features to CSV. Returns the resolved path."""
    target = Path(path) if path else (PROCESSED_DIR / "processed_features.csv")
    ensure_dir(target.parent)
    df.to_csv(target, index=False)
    return target


def full_preprocess(
    raw_path: str | Path | None = None,
    *,
    seed: int = 17,
) -> pd.DataFrame:
    """Convenience: load → clean → sort → validate → impute. Returns the
    processed frame. Does *not* write to disk; that is the caller's job.
    """
    df = load_raw_or_synthetic_data(raw_path)
    df = clean_data(df)
    df = sort_by_participant_date(df)
    validate_cohort(df, raise_on_error=True)
    df = impute_safe_defaults(df)
    return df
