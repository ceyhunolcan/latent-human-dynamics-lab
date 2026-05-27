"""Garmin Connect adapter.

Garmin Connect's CSV export gives one row per day with columns named
roughly like:

* Date
* Steps
* Total Calories (BMR + Active)
* Resting HR
* Average HR
* Max HR
* Avg Stress
* Min HRV (ms)
* Sleep Duration (h:m)

The adapter is tolerant to small label variations because Garmin has
shipped multiple export formats over the years.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


SCHEMA_MAPPING: dict[str, str] = {
    "Date": "date",
    "Steps": "daily_steps",
    "Resting HR": "resting_hr",
    "Resting Heart Rate": "resting_hr",
    "Average Stress": "stress_score",
    "Avg Stress": "stress_score",
    "Min HRV": "hrv_rmssd",
    "Avg HRV": "hrv_rmssd",
    "Sleep Duration": "sleep_duration",
    "Active Minutes": "active_minutes",
    "Intensity Minutes (Moderate)": "active_minutes",
}


def _parse_duration_hms(value: object) -> float:
    """Parse '7h 12m' or '07:12' into hours. Returns NaN on failure."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    s = str(value).strip()
    m_hm = re.match(r"^\s*(\d+)\s*h\s*(\d+)?\s*m?\s*$", s)
    if m_hm:
        h = int(m_hm.group(1))
        mn = int(m_hm.group(2) or 0)
        return h + mn / 60.0
    m_colon = re.match(r"^\s*(\d+):(\d+)\s*$", s)
    if m_colon:
        return int(m_colon.group(1)) + int(m_colon.group(2)) / 60.0
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_garmin(
    csv_path: str | Path,
    participant_id: str = "P_garmin_user",
) -> pd.DataFrame:
    """Load a Garmin Connect CSV.

    Parameters
    ----------
    csv_path : path
        Path to the per-day CSV exported from Garmin Connect.
    participant_id : str
        Garmin CSVs do not carry a participant identifier; assign one.

    Returns
    -------
    DataFrame
        Long-format daily frame with canonical columns where they exist.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Garmin CSV not found at {p!s}. "
            "Export from Garmin Connect → Settings → Account → Export Data."
        )
    raw = pd.read_csv(p)
    out_cols: dict[str, pd.Series] = {}
    for src_col, canonical in SCHEMA_MAPPING.items():
        if src_col in raw.columns:
            if canonical == "sleep_duration":
                out_cols[canonical] = raw[src_col].map(_parse_duration_hms)
            else:
                out_cols[canonical] = pd.to_numeric(raw[src_col], errors="coerce")
    out = pd.DataFrame(out_cols)
    if "date" not in out.columns and "Date" in raw.columns:
        out["date"] = pd.to_datetime(raw["Date"], errors="coerce")
    out.insert(0, "participant_id", participant_id)
    return out
