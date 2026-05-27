"""Structured summary of a pipeline run.

Tracks per-stage metrics (rows in/out, NaN rate, timing) so a pipeline
operator has an audit trail. Used by `scripts/run_pipeline.py` and is
useful for any caller that wants to verify their cohort flowed through
the stages correctly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StageReport:
    """One stage of the pipeline."""
    name: str
    rows_in: int
    rows_out: int
    columns_in: int
    columns_out: int
    nan_rate_in: float
    nan_rate_out: float
    duration_seconds: float
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineSummary:
    """Sequence of `StageReport`s with helpers for serialization + display."""
    stages: List[StageReport] = field(default_factory=list)

    def record(self, name: str, df_in: pd.DataFrame, df_out: pd.DataFrame,
               duration_seconds: float, notes: str = "") -> None:
        """Append a stage report computed from two DataFrame snapshots."""
        self.stages.append(StageReport(
            name=name,
            rows_in=len(df_in),
            rows_out=len(df_out),
            columns_in=df_in.shape[1] if df_in.ndim == 2 else 0,
            columns_out=df_out.shape[1] if df_out.ndim == 2 else 0,
            nan_rate_in=_nan_rate(df_in),
            nan_rate_out=_nan_rate(df_out),
            duration_seconds=duration_seconds,
            notes=notes,
        ))

    def as_dict(self) -> Dict[str, Any]:
        return {"stages": [s.as_dict() for s in self.stages]}

    def as_markdown(self) -> str:
        if not self.stages:
            return "_(no stages recorded)_"
        lines = [
            "| Stage | Rows | Cols | NaN% | Duration |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for s in self.stages:
            row_change = f"{s.rows_in}→{s.rows_out}"
            col_change = f"{s.columns_in}→{s.columns_out}"
            nan_change = f"{100 * s.nan_rate_in:.1f}→{100 * s.nan_rate_out:.1f}"
            dur = f"{s.duration_seconds:.2f}s"
            lines.append(f"| {s.name} | {row_change} | {col_change} | {nan_change} | {dur} |")
        return "\n".join(lines)


def _nan_rate(df: pd.DataFrame) -> float:
    """Fraction of numeric cells that are NaN. Returns 0 for empty DataFrames."""
    if len(df) == 0:
        return 0.0
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return 0.0
    total = numeric.size
    if total == 0:
        return 0.0
    return float(numeric.isna().sum().sum() / total)


class StageTimer:
    """Context manager that times a block. Pair with `PipelineSummary.record`.

    Usage::

        timer = StageTimer()
        df_out = transform(df_in)
        summary.record("feature engineering", df_in, df_out, timer.elapsed())
    """

    def __init__(self):
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start
