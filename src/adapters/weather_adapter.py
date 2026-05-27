"""Weather / environment adapter.

Reads a daily-resolution CSV (NOAA GHCN, Open-Meteo, or any similar
source) and coerces it to the canonical environmental columns used by
the pipeline.

The adapter does not make network calls. The intended workflow is for a
user to pre-download daily summaries to ``data/raw/weather_<location>.csv``
and feed the path here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_MAPPING: dict[str, str] = {
    "date": "date",
    "DATE": "date",
    "tavg": "temperature_c",
    "tmean": "temperature_c",
    "temperature_2m_mean": "temperature_c",
    "tmin": "nighttime_temperature_c",
    "temperature_2m_min": "nighttime_temperature_c",
    "tmax": "temperature_c_max",
    "rhum": "humidity",
    "relative_humidity_2m_mean": "humidity",
    "humidity": "humidity",
    "aqi": "aqi",
    "us_aqi": "aqi",
}


def _compute_heat_index(temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
    """A simple heat-index approximation in degrees C."""
    return temp_c + 0.07 * (humidity - 50).clip(lower=0)


def _flag_heat_wave(heat_index: pd.Series, threshold: float = 32.0, min_run: int = 3) -> pd.Series:
    """Mark 3+ consecutive days with heat index above threshold."""
    over = heat_index > threshold
    run = (over != over.shift()).cumsum()
    sizes = over.groupby(run).transform("size")
    return ((over) & (sizes >= min_run)).astype(int)


def load_weather(csv_path: str | Path) -> pd.DataFrame:
    """Load a daily weather CSV and return canonical environmental columns.

    Parameters
    ----------
    csv_path : path
        Path to a CSV with at minimum a date column and either daily mean
        temperature or daily min/max temperatures.

    Returns
    -------
    DataFrame
        Columns: ``date``, ``temperature_c``, ``nighttime_temperature_c``,
        ``humidity``, ``aqi``, ``heat_index``, ``heat_wave_flag``,
        ``poor_air_quality_flag``. Missing inputs become NaN; the
        downstream preprocessor handles them.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Weather CSV not found at {p!s}. "
            "Download daily summaries (NOAA GHCN, Open-Meteo, etc.) "
            "to data/raw/ and pass that path here."
        )
    raw = pd.read_csv(p)
    cols: dict[str, pd.Series] = {}
    for src, dest in SCHEMA_MAPPING.items():
        if src in raw.columns:
            cols.setdefault(dest, raw[src])
    out = pd.DataFrame(cols)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Derive heat index when humidity is known
    if "temperature_c" in out and "humidity" in out:
        out["heat_index"] = _compute_heat_index(
            pd.to_numeric(out["temperature_c"], errors="coerce"),
            pd.to_numeric(out["humidity"], errors="coerce"),
        )
        out["heat_wave_flag"] = _flag_heat_wave(out["heat_index"])
    else:
        out["heat_index"] = np.nan
        out["heat_wave_flag"] = 0

    if "aqi" in out:
        out["poor_air_quality_flag"] = (
            pd.to_numeric(out["aqi"], errors="coerce") > 150
        ).astype(int)
    else:
        out["aqi"] = np.nan
        out["poor_air_quality_flag"] = 0

    return out
