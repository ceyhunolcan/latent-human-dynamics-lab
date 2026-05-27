"""Subgroup performance reporting.

A thin wrapper that takes a cohort dataframe, a list of subgrouping columns,
and a predict function, and returns a tidy table of per-subgroup metrics.
The intent is fairness/equity auditing: we want to make sure that any
average improvements do not hide systematic underperformance in a subgroup.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .metrics import classification_metrics


def subgroup_performance_table(
    df: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    predict_fn: Callable,
    grouping_cols: Iterable,
    min_subgroup_size: int = 25,
) -> pd.DataFrame:
    """Return one row per non-trivial subgroup with classification metrics."""
    rows = []
    for col in grouping_cols:
        if col not in df.columns:
            continue
        # Coerce to category labels (small number of unique values).
        groups = df[col]
        if pd.api.types.is_numeric_dtype(groups) and groups.nunique() > 6:
            # Bin numeric subgroupings into tertiles.
            try:
                groups = pd.qcut(groups, q=3, labels=["low", "mid", "high"], duplicates="drop")
            except ValueError:
                continue
        for label in pd.unique(groups.dropna()):
            mask = groups == label
            sub = df[mask]
            if len(sub) < min_subgroup_size:
                continue
            y = sub[y_col].to_numpy()
            y_pred = predict_fn(sub[feature_cols].to_numpy())
            m = classification_metrics(y, y_pred)
            m["grouping"] = col
            m["subgroup"] = str(label)
            m["n"] = int(len(sub))
            rows.append(m)
    return pd.DataFrame(rows)
