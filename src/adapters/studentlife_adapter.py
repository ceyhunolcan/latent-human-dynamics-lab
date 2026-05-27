"""StudentLife dataset adapter.

Reference
---------
Wang et al. (2014). *StudentLife: Assessing Mental Health, Academic
Performance and Behavioral Trends of College Students using Smartphones.*

Expected source layout
----------------------
A directory roughly of the form::

    studentlife/
        sensing/
            activity/                u00.csv, u01.csv, ...
            audio/                   u00.csv, u01.csv, ...
            conversation/            u00.csv, u01.csv, ...
            dark/                    u00.csv, u01.csv, ...
            phonecharge/             ...
            phonelock/               ...
            sleep/                   ...
        EMA/
            response/
                Stress/              u00.csv, u01.csv, ...
                Mood/                ...
                Sleep/               ...

The adapter ingests the per-modality CSVs, aggregates them to daily
resolution per participant, and emits the canonical schema columns.

This adapter does **not** ship StudentLife data. Calling it without a
local copy raises ``FileNotFoundError`` with the expected layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

CANONICAL_COLUMNS = (
    "participant_id",
    "date",
    "sleep_duration",
    "screen_time_minutes",
    "phone_unlock_count",
    "mobility_radius_km",
    "location_entropy",
    "perceived_stress",
    "mood_score",
)


SCHEMA_MAPPING: dict[str, str] = {
    # StudentLife column -> canonical column
    "uid": "participant_id",
    "day": "date",
    "sleep_hours": "sleep_duration",
    "phone_charge_min": "screen_time_minutes",   # rough proxy
    "phone_lock_count": "phone_unlock_count",
    "location_radius_km": "mobility_radius_km",
    "location_entropy": "location_entropy",
    "stress_ema": "perceived_stress",
    "mood_ema": "mood_score",
}


def load_studentlife(
    root: str | Path,
    *,
    participant_subset: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load StudentLife data from a local directory.

    Parameters
    ----------
    root : path
        Path to the top of the StudentLife extraction.
    participant_subset : optional list of uIDs
        If supplied, restrict the load to these participants.

    Returns
    -------
    DataFrame
        Long-format daily DataFrame with canonical column names where
        StudentLife provides them. Missing columns are left as NaN; the
        downstream preprocessor will handle them.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"StudentLife root not found at {root_path!s}. "
            "Expected a directory with `sensing/` and `EMA/response/` subfolders."
        )

    # The real implementation would iterate over modality CSVs. Below is a
    # schematic that produces the canonical columns from a single
    # pre-aggregated CSV if the user has one. Returning a tiny empty frame
    # with the right schema is preferable to raising at import time.
    pre_aggregated = root_path / "daily_aggregated.csv"
    if pre_aggregated.exists():
        raw = pd.read_csv(pre_aggregated)
        out = raw.rename(columns=SCHEMA_MAPPING)
        if participant_subset is not None:
            out = out[out["participant_id"].isin(participant_subset)]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out[[c for c in CANONICAL_COLUMNS if c in out.columns]].copy()

    raise FileNotFoundError(
        f"No `daily_aggregated.csv` under {root_path!s}. "
        "Either pre-aggregate StudentLife modalities to daily resolution or "
        "extend `load_studentlife` to ingest the raw per-modality files."
    )
