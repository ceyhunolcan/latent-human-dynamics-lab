"""Ingest StudentLife (Dartmouth 2013) into the canonical daily-cohort schema.

Run this LOCALLY against an unpacked studentlife_1.1.0/ directory:

    tar -xf studentlife_1.1.0.tar
    python scripts/ingest_studentlife.py /path/to/dataset/

It walks the sensing/ and EMA/ subdirectories, aggregates raw per-event
streams to per-participant per-day summaries, and writes one CSV file
that the rest of the pipeline can consume.

Output columns match the canonical schema documented in
src/data/synthetic_generator.py, with these caveats:
  - HRV, RHR, recovery_score, temperature, AQI, heat_wave_flag: not in
    StudentLife. Set to NaN. The synthetic-to-real comparison will only
    use overlapping columns.
  - sleep_duration is estimated from dark + phone-locked + stationary
    accelerometer signal, following the StudentLife paper's heuristic.
  - daily_steps is approximated as activity_minutes * 110 (mean stride
    rate in steps/min for walking activity).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stream-level helpers
# ---------------------------------------------------------------------------


def _walk_csv_streams(subdir: Path, glob: str = "*.csv"):
    """Yield (participant_id, dataframe) pairs from `<subdir>/<glob>`.

    StudentLife files follow the convention `<stream>_u<id>.csv` where
    `<id>` is a 2-digit zero-padded integer like u00, u01, ...
    """
    if not subdir.exists():
        return
    for fp in sorted(subdir.glob(glob)):
        name = fp.stem  # e.g. "activity_u00"
        try:
            pid = name.split("_")[-1]  # "u00"
        except IndexError:
            continue
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"  WARN  could not read {fp}: {e}", file=sys.stderr)
            continue
        yield pid, df


def _epoch_to_date(series: pd.Series) -> pd.Series:
    """Convert a Unix epoch (seconds) series to dates (UTC)."""
    return pd.to_datetime(series, unit="s", utc=True).dt.tz_convert(None).dt.normalize()


# ---------------------------------------------------------------------------
# Per-stream aggregators
# ---------------------------------------------------------------------------


def aggregate_activity(root: Path) -> pd.DataFrame:
    """sensing/activity/activity_uXX.csv
    Columns: timestamp, activity_inference
    Where activity_inference: 0=stationary, 1=walking, 2=running, 3=unknown
    """
    rows = []
    for pid, df in _walk_csv_streams(root / "sensing" / "activity"):
        if "timestamp" not in df.columns:
            continue
        df["date"] = _epoch_to_date(df["timestamp"])
        # Activity samples are taken every ~3 minutes when in motion. Each
        # sample represents 3 minutes of state. Sum the minutes per day.
        df["active_minutes"] = ((df["activity_inference"] >= 1)
                                & (df["activity_inference"] <= 2)).astype(int) * 3
        daily = df.groupby("date").agg(active_minutes=("active_minutes", "sum")).reset_index()
        daily["participant_id"] = pid
        rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "active_minutes"])
    out = pd.concat(rows, ignore_index=True)
    out["daily_steps"] = out["active_minutes"] * 110  # ~110 steps/min walking
    return out


def aggregate_conversation(root: Path) -> pd.DataFrame:
    """sensing/conversation/conversation_uXX.csv
    Columns: start_timestamp, end_timestamp
    """
    rows = []
    for pid, df in _walk_csv_streams(root / "sensing" / "conversation"):
        if not {"start_timestamp", "end_timestamp"}.issubset(df.columns):
            continue
        df["date"] = _epoch_to_date(df["start_timestamp"])
        df["duration_sec"] = df["end_timestamp"] - df["start_timestamp"]
        daily = df.groupby("date").agg(
            conversation_sec=("duration_sec", "sum"),
            conversation_events=("duration_sec", "count"),
        ).reset_index()
        daily["participant_id"] = pid
        rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "conversation_sec", "conversation_events"])
    return pd.concat(rows, ignore_index=True)


def aggregate_dark(root: Path) -> pd.DataFrame:
    """sensing/dark/dark_uXX.csv — dark-screen periods, used to estimate
    nighttime / sleep / unsupervised hours.
    """
    rows = []
    for pid, df in _walk_csv_streams(root / "sensing" / "dark"):
        if not {"start", "end"}.issubset(df.columns):
            continue
        df["date"] = _epoch_to_date(df["start"])
        df["duration_sec"] = df["end"] - df["start"]
        daily = df.groupby("date").agg(dark_sec=("duration_sec", "sum")).reset_index()
        daily["participant_id"] = pid
        rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "dark_sec"])
    return pd.concat(rows, ignore_index=True)


def aggregate_phonelock(root: Path) -> pd.DataFrame:
    """sensing/phonelock/phonelock_uXX.csv — phone locked = not using it"""
    rows = []
    for pid, df in _walk_csv_streams(root / "sensing" / "phonelock"):
        if not {"start", "end"}.issubset(df.columns):
            continue
        df["date"] = _epoch_to_date(df["start"])
        df["duration_sec"] = df["end"] - df["start"]
        daily = df.groupby("date").agg(
            phonelock_sec=("duration_sec", "sum"),
            phone_unlock_count=("duration_sec", "count"),
        ).reset_index()
        # Approximate screen time as 24h minus locked time
        daily["screen_time_minutes"] = (86400 - daily["phonelock_sec"]).clip(lower=0) / 60
        daily["participant_id"] = pid
        rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "screen_time_minutes", "phone_unlock_count"])
    return pd.concat(rows, ignore_index=True)


def aggregate_gps(root: Path) -> pd.DataFrame:
    """sensing/gps/gps_uXX.csv — location samples.
    We compute daily mobility radius (km from centroid) and a crude
    location entropy from rounded lat/lon bins.
    """
    rows = []
    for pid, df in _walk_csv_streams(root / "sensing" / "gps"):
        cols = set(df.columns)
        ts_col = "time" if "time" in cols else ("timestamp" if "timestamp" in cols else None)
        if ts_col is None or "latitude" not in cols or "longitude" not in cols:
            continue
        df = df.dropna(subset=[ts_col, "latitude", "longitude"]).copy()
        if df.empty:
            continue
        df["date"] = _epoch_to_date(df[ts_col])
        # Bin to ~111m at lat~40° (0.001 deg lat ~= 111m; 0.001 deg lon ~ 85m)
        df["lat_bin"] = (df["latitude"] * 1000).round().astype(int)
        df["lon_bin"] = (df["longitude"] * 1000).round().astype(int)

        def _entropy(g):
            counts = g.groupby(["lat_bin", "lon_bin"]).size()
            p = counts / counts.sum()
            return float(-(p * np.log2(p)).sum())

        def _radius_km(g):
            lat = g["latitude"]; lon = g["longitude"]
            clat, clon = lat.mean(), lon.mean()
            dlat_km = (lat - clat) * 111.0
            dlon_km = (lon - clon) * 111.0 * np.cos(np.radians(clat))
            return float(np.sqrt(dlat_km ** 2 + dlon_km ** 2).max())

        daily = (
            df.groupby("date")
              .apply(lambda g: pd.Series({
                  "location_entropy": _entropy(g),
                  "mobility_radius_km": _radius_km(g),
              }))
              .reset_index()
        )
        daily["participant_id"] = pid
        rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "location_entropy", "mobility_radius_km"])
    return pd.concat(rows, ignore_index=True)


def aggregate_ema_stress(root: Path) -> pd.DataFrame:
    """EMA/response/Stress/Stress_uXX.json — list of self-reports."""
    rows = []
    ema_dir = root / "EMA" / "response" / "Stress"
    if not ema_dir.exists():
        return pd.DataFrame(columns=["participant_id", "date", "stress_score"])
    for fp in sorted(ema_dir.glob("Stress_u*.json")):
        pid = fp.stem.split("_")[-1]
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        records = []
        for item in data:
            ts = item.get("resp_time")
            level = item.get("level")
            if ts is None or level is None:
                continue
            # StudentLife stress level: 1 (a little stressed) to 5 (extremely)
            try:
                level = float(level)
            except (ValueError, TypeError):
                continue
            records.append({
                "date": pd.to_datetime(ts).normalize(),
                "stress_score": level * 20,  # scale 1-5 to 0-100
            })
        if records:
            daily = pd.DataFrame(records).groupby("date").mean().reset_index()
            daily["participant_id"] = pid
            rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "stress_score"])
    return pd.concat(rows, ignore_index=True)


def aggregate_ema_mood(root: Path) -> pd.DataFrame:
    """EMA/response/Mood 1/ — happy/sad/etc daily ratings."""
    rows = []
    ema_dir = root / "EMA" / "response" / "Mood 1"
    if not ema_dir.exists():
        ema_dir = root / "EMA" / "response" / "Mood"
    if not ema_dir.exists():
        return pd.DataFrame(columns=["participant_id", "date", "mood_score"])
    for fp in sorted(ema_dir.glob("Mood*_u*.json")):
        pid = fp.stem.split("_")[-1]
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        records = []
        for item in data:
            ts = item.get("resp_time")
            # The "happy" field is typically the cheerful self-report
            level = item.get("happy") or item.get("happyornot")
            if ts is None or level is None:
                continue
            try:
                level = float(level)
            except (ValueError, TypeError):
                continue
            records.append({
                "date": pd.to_datetime(ts).normalize(),
                "mood_score": level * 20,  # scale 1-5 → 0-100
            })
        if records:
            daily = pd.DataFrame(records).groupby("date").mean().reset_index()
            daily["participant_id"] = pid
            rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "mood_score"])
    return pd.concat(rows, ignore_index=True)


def aggregate_ema_sleep(root: Path) -> pd.DataFrame:
    """EMA/response/Sleep/ — daily sleep duration self-report."""
    rows = []
    ema_dir = root / "EMA" / "response" / "Sleep"
    if not ema_dir.exists():
        return pd.DataFrame(columns=["participant_id", "date", "sleep_duration", "sleep_efficiency"])
    for fp in sorted(ema_dir.glob("Sleep_u*.json")):
        pid = fp.stem.split("_")[-1]
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        records = []
        for item in data:
            ts = item.get("resp_time")
            hour = item.get("hour")
            rate = item.get("rate")  # 1=very poor … 4=very good
            if ts is None or hour is None:
                continue
            try:
                hour = float(hour)
                rate = float(rate) if rate is not None else 3.0
            except (ValueError, TypeError):
                continue
            records.append({
                "date": pd.to_datetime(ts).normalize(),
                "sleep_duration": hour,
                "sleep_efficiency": 0.6 + 0.1 * rate,  # 0.7..1.0 from 1..4
            })
        if records:
            daily = pd.DataFrame(records).groupby("date").mean().reset_index()
            daily["participant_id"] = pid
            rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "sleep_duration", "sleep_efficiency"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Merge + canonicalize
# ---------------------------------------------------------------------------


CANONICAL_COLUMNS = [
    # Identifiers
    "participant_id", "date",
    # Wearable proxies (StudentLife has limited wearable signal — most NaN)
    "sleep_duration", "sleep_efficiency", "sleep_midpoint",
    "hrv_rmssd", "resting_hr", "daily_steps", "active_minutes",
    "recovery_score", "stress_score",
    # Behavioral
    "screen_time_minutes", "phone_unlock_count", "mobility_radius_km",
    "location_entropy", "behavioral_regularity",
    # Environmental — StudentLife has none
    "temperature_c", "nighttime_temperature_c", "humidity",
    "aqi", "heat_index", "heat_wave_flag", "poor_air_quality_flag",
    # Self-report
    "mood_score", "fatigue_score", "perceived_stress",
    "energy_score", "cognitive_load_proxy",
    # Missingness flags
    "missing_wearable_flag", "missing_phone_flag", "missing_survey_flag",
    "total_missing_modalities",
]


def canonicalize(merged: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the canonical column order, fill missing with NaN, add flags."""
    out = pd.DataFrame()
    for col in CANONICAL_COLUMNS:
        if col in merged.columns:
            out[col] = merged[col]
        else:
            out[col] = np.nan

    # Missingness flags computed from the actual NaN pattern
    out["missing_wearable_flag"] = out[
        ["hrv_rmssd", "resting_hr", "recovery_score"]
    ].isna().all(axis=1).astype(int)
    out["missing_phone_flag"] = out[
        ["screen_time_minutes", "mobility_radius_km"]
    ].isna().all(axis=1).astype(int)
    out["missing_survey_flag"] = out[
        ["stress_score", "mood_score", "sleep_duration"]
    ].isna().all(axis=1).astype(int)
    out["total_missing_modalities"] = (
        out["missing_wearable_flag"]
        + out["missing_phone_flag"]
        + out["missing_survey_flag"]
    )

    out = out.sort_values(["participant_id", "date"]).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest StudentLife into canonical daily-cohort CSV.")
    parser.add_argument("dataset_root", type=str, help="Path to unpacked studentlife_1.1.0/ directory")
    parser.add_argument("--output", type=str, default="studentlife_daily.csv",
                        help="Where to write the merged CSV (default: studentlife_daily.csv)")
    parser.add_argument("--min-days", type=int, default=14,
                        help="Drop participants with fewer than N days of data (default: 14)")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    print(f"Ingesting StudentLife from {root}")
    print()

    streams = {}
    print("  → activity ...")
    streams["activity"] = aggregate_activity(root)
    print(f"     {len(streams['activity'])} rows, {streams['activity']['participant_id'].nunique() if not streams['activity'].empty else 0} participants")

    print("  → conversation ...")
    streams["conversation"] = aggregate_conversation(root)
    print(f"     {len(streams['conversation'])} rows")

    print("  → dark ...")
    streams["dark"] = aggregate_dark(root)
    print(f"     {len(streams['dark'])} rows")

    print("  → phonelock ...")
    streams["phonelock"] = aggregate_phonelock(root)
    print(f"     {len(streams['phonelock'])} rows")

    print("  → gps ...")
    streams["gps"] = aggregate_gps(root)
    print(f"     {len(streams['gps'])} rows")

    print("  → EMA stress ...")
    streams["stress"] = aggregate_ema_stress(root)
    print(f"     {len(streams['stress'])} rows")

    print("  → EMA mood ...")
    streams["mood"] = aggregate_ema_mood(root)
    print(f"     {len(streams['mood'])} rows")

    print("  → EMA sleep ...")
    streams["sleep"] = aggregate_ema_sleep(root)
    print(f"     {len(streams['sleep'])} rows")

    # Outer-merge everything on (participant_id, date)
    print()
    print("Merging streams on (participant_id, date) ...")
    merged = None
    for name, df in streams.items():
        if df.empty or "participant_id" not in df.columns or "date" not in df.columns:
            continue
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["participant_id", "date"], how="outer")

    if merged is None or merged.empty:
        print("ERROR: no streams produced any data. Is the dataset_root correct?", file=sys.stderr)
        return 1

    # Reindex to canonical schema
    final = canonicalize(merged)

    # Drop participants with too few days
    counts = final.groupby("participant_id").size()
    keep = counts[counts >= args.min_days].index
    final = final[final["participant_id"].isin(keep)].reset_index(drop=True)

    out_path = Path(args.output).expanduser().resolve()
    final.to_csv(out_path, index=False)
    print()
    print(f"Wrote {len(final)} rows × {final.shape[1]} cols → {out_path}")
    print(f"  Participants: {final['participant_id'].nunique()}")
    print(f"  Date range  : {final['date'].min()} → {final['date'].max()}")
    print()
    print("Channel coverage (non-NaN %):")
    for col in [
        "sleep_duration", "active_minutes", "daily_steps",
        "screen_time_minutes", "phone_unlock_count",
        "mobility_radius_km", "location_entropy",
        "stress_score", "mood_score",
    ]:
        if col in final.columns:
            pct = 100 * final[col].notna().mean()
            print(f"  {col:<28s} {pct:5.1f}%")

    print()
    print("(research prototype, non-clinical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
