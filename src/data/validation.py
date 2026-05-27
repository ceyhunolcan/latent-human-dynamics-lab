"""Schema and value-range validation for the synthetic and real cohort frames.

This is deliberately strict: a failed validation halts the pipeline.
Real-data adapters are expected to coerce their outputs into the schema
defined here before passing to downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when a cohort frame fails one of the validation checks."""


REQUIRED_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "date",
    "age",
    "sex",
    "chronotype",
    "sleep_duration",
    "sleep_efficiency",
    "sleep_midpoint",
    "hrv_rmssd",
    "resting_hr",
    "daily_steps",
    "active_minutes",
    "screen_time_minutes",
    "phone_unlock_count",
    "temperature_c",
    "nighttime_temperature_c",
    "humidity",
    "aqi",
    "heat_wave_flag",
    "missing_wearable_flag",
    "missing_phone_flag",
    "missing_survey_flag",
    "regime_label",
)


# Each entry: column → (lower bound, upper bound) on **non-NaN** values.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "sleep_duration": (0, 16),
    "sleep_efficiency": (0.0, 1.0),
    "sleep_midpoint": (-2, 14),  # allow wraparound representations
    "hrv_rmssd": (1, 250),
    "resting_hr": (25, 220),
    "daily_steps": (0, 100_000),
    "active_minutes": (0, 600),
    "screen_time_minutes": (0, 1440),
    "phone_unlock_count": (0, 2000),
    "temperature_c": (-50, 60),
    "nighttime_temperature_c": (-50, 50),
    "humidity": (0, 100),
    "aqi": (0, 1000),
    "mood_score": (0, 100),
    "fatigue_score": (0, 100),
    "perceived_stress": (0, 100),
}


BINARY_FLAG_COLS: tuple[str, ...] = (
    "heat_wave_flag",
    "poor_air_quality_flag",
    "missing_wearable_flag",
    "missing_phone_flag",
    "missing_survey_flag",
)


@dataclass
class ValidationReport:
    """Outcome of running :func:`validate_cohort`."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"Missing required columns: {missing}"]
    return []


def _check_dates(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    try:
        parsed = pd.to_datetime(df["date"], errors="raise")
    except (ValueError, TypeError) as e:
        errors.append(f"Date column failed to parse: {e}")
        return errors
    if parsed.isna().any():
        errors.append("Date column contains NaT after parsing.")
    return errors


def _check_uniqueness(df: pd.DataFrame) -> list[str]:
    if df.duplicated(subset=["participant_id", "date"]).any():
        n_dups = int(df.duplicated(subset=["participant_id", "date"]).sum())
        return [f"Duplicate (participant_id, date) rows: {n_dups}"]
    return []


def _check_numeric_ranges(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((s < lo) | (s > hi)) & s.notna()
        n = int(out_of_range.sum())
        if n > 0:
            errors.append(
                f"Column {col!r}: {n} values outside [{lo}, {hi}] "
                f"(observed [{float(s.min()):.2f}, {float(s.max()):.2f}])."
            )
    return errors


def _check_binary_flags(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for col in BINARY_FLAG_COLS:
        if col not in df.columns:
            continue
        vals = pd.unique(df[col].dropna())
        bad = [v for v in vals if v not in (0, 1, 0.0, 1.0, True, False)]
        if bad:
            errors.append(f"Column {col!r}: non-binary values {bad}.")
    return errors


def _check_regime_labels(df: pd.DataFrame) -> list[str]:
    if "regime_label" not in df.columns:
        return []
    allowed = {"stable", "stressed", "dysregulated", "recovery"}
    bad = set(pd.unique(df["regime_label"].dropna())) - allowed
    if bad:
        return [f"regime_label has unexpected values: {sorted(bad)} (allowed: {sorted(allowed)})."]
    return []


def validate_cohort(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Run all validation checks and return a report.

    If ``raise_on_error`` is True (default) and there are any errors,
    raises :class:`ValidationError` with the joined messages.
    """
    errors: list[str] = []
    errors.extend(_check_required_columns(df, required_columns))
    if errors:
        # Skip downstream checks; columns are missing.
        report = ValidationReport(ok=False, errors=errors)
    else:
        errors.extend(_check_dates(df))
        errors.extend(_check_uniqueness(df))
        errors.extend(_check_numeric_ranges(df))
        errors.extend(_check_binary_flags(df))
        errors.extend(_check_regime_labels(df))
        report = ValidationReport(ok=(len(errors) == 0), errors=errors)

    # Warnings: high NaN fractions in critical columns
    warnings: list[str] = []
    for col in ("hrv_rmssd", "sleep_duration", "screen_time_minutes"):
        if col in df.columns:
            frac = float(df[col].isna().mean())
            if frac > 0.6:
                warnings.append(f"Column {col!r}: NaN fraction {frac:.2%} is unusually high.")
    report.warnings = warnings

    if raise_on_error and not report.ok:
        raise ValidationError("\n".join(report.errors))

    return report
