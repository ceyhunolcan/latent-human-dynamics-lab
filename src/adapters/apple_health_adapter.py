"""Apple Health export adapter.

Apple Health exports as a zipped ``export.xml``. The adapter reads the
XML, picks the records we care about (sleep, HRV, resting HR, step count,
active minutes), aggregates to daily resolution, and emits canonical
columns.

This adapter does **not** ship Apple Health data. Calling it without a
local copy raises ``FileNotFoundError`` with the expected layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

# Apple Health record types we know how to handle.
TYPE_MAP: dict[str, str] = {
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_rmssd",  # SDNN proxy
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
    "HKQuantityTypeIdentifierStepCount": "daily_steps",
    "HKQuantityTypeIdentifierAppleExerciseTime": "active_minutes",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep_duration",
}


def _iter_records(xml_path: Path) -> Iterator[dict]:
    """Stream-parse the Apple Health export XML."""
    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag == "Record":
            yield dict(elem.attrib)
            elem.clear()


def load_apple_health(
    export_xml: str | Path,
    participant_id: str = "P_apple_user",
) -> pd.DataFrame:
    """Load an Apple Health export and aggregate to daily rows.

    Parameters
    ----------
    export_xml : path
        Path to ``export.xml`` (the unzipped Apple Health export).
    participant_id : str
        Apple exports do not carry a participant identifier; assign one.

    Returns
    -------
    DataFrame
        Long-format daily DataFrame with canonical columns.
    """
    p = Path(export_xml)
    if not p.exists():
        raise FileNotFoundError(
            f"Apple Health export.xml not found at {p!s}. "
            "Unzip the export from Settings → Health → Export All Health Data."
        )

    rows: list[dict] = []
    for rec in _iter_records(p):
        t = rec.get("type", "")
        if t not in TYPE_MAP:
            continue
        canonical = TYPE_MAP[t]
        try:
            start = pd.to_datetime(rec.get("startDate"))
            end = pd.to_datetime(rec.get("endDate", start))
        except (TypeError, ValueError):
            continue
        if t == "HKCategoryTypeIdentifierSleepAnalysis":
            value = max(0.0, (end - start).total_seconds() / 3600.0)
        else:
            try:
                value = float(rec.get("value", "nan"))
            except (TypeError, ValueError):
                continue
        rows.append({"date": start.normalize(), "column": canonical, "value": value})

    if not rows:
        return pd.DataFrame(columns=["participant_id", "date"])

    long_df = pd.DataFrame(rows)
    agg = long_df.groupby(["date", "column"], as_index=False)["value"].agg(
        lambda s: float(np.nanmean(s)) if s.notna().any() else np.nan
    )
    wide = agg.pivot(index="date", columns="column", values="value").reset_index()
    wide.insert(0, "participant_id", participant_id)
    wide.columns.name = None
    return wide
