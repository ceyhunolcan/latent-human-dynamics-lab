"""Robustness diagnostics.

Each function takes a fitted predictor and an evaluation cohort, perturbs
the inputs along a specific dimension (missingness rate, heat-wave exposure,
climate vulnerability, environmental shock magnitude), and reports how
performance changes. The intent is to surface failure modes before they
appear in deployment, not to certify performance.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .metrics import classification_metrics, regression_metrics


def missingness_stress_test(
    df: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    predict_fn: Callable,
    missing_rates: tuple = (0.0, 0.1, 0.2, 0.4, 0.6),
    seed: int = 17,
) -> pd.DataFrame:
    """Inject random missingness into feature columns and measure degradation.

    `predict_fn(X)` must return either class predictions or a (mean,) tuple.
    Imputation between dropping and prediction is zero-fill, intentionally
    naive: the point is to see how the *model* responds to dropout, not how
    a complex imputation pipeline can paper over it.
    """
    rng = np.random.default_rng(seed)
    results = []
    y = df[y_col].to_numpy()
    X = df[feature_cols].to_numpy()
    for rate in missing_rates:
        Xp = X.copy()
        mask = rng.random(Xp.shape) < rate
        Xp[mask] = 0.0
        y_pred = predict_fn(Xp)
        m = classification_metrics(y, y_pred)
        m["missingness_rate"] = float(rate)
        results.append(m)
    return pd.DataFrame(results)


def heatwave_subgroup_analysis(
    df: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    predict_fn: Callable,
    heat_col: str = "heat_wave_flag",
) -> pd.DataFrame:
    """Compare performance during heat-wave days vs. non-heat-wave days."""
    rows = []
    for subgroup_name, mask in [
        ("non_heatwave", df[heat_col] == 0),
        ("heatwave", df[heat_col] == 1),
    ]:
        sub = df[mask]
        if len(sub) < 20:
            continue
        y = sub[y_col].to_numpy()
        y_pred = predict_fn(sub[feature_cols].to_numpy())
        m = classification_metrics(y, y_pred)
        m["subgroup"] = subgroup_name
        m["n"] = int(len(sub))
        rows.append(m)
    return pd.DataFrame(rows)


def climate_vulnerability_subgroup_analysis(
    df: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    predict_fn: Callable,
    vuln_col: str = "baseline_climate_vulnerability",
) -> pd.DataFrame:
    """Tertile split on climate vulnerability; report metrics per tertile."""
    if vuln_col not in df.columns:
        return pd.DataFrame(
            [{"subgroup": "missing", "note": f"{vuln_col} not in dataframe"}]
        )
    tertiles = pd.qcut(df[vuln_col], q=3, labels=["low", "mid", "high"], duplicates="drop")
    rows = []
    for label in tertiles.cat.categories:
        sub = df[tertiles == label]
        if len(sub) < 20:
            continue
        y = sub[y_col].to_numpy()
        y_pred = predict_fn(sub[feature_cols].to_numpy())
        m = classification_metrics(y, y_pred)
        m["subgroup"] = f"vulnerability_{label}"
        m["n"] = int(len(sub))
        rows.append(m)
    return pd.DataFrame(rows)


def out_of_distribution_environmental_shock_test(
    df: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    predict_fn: Callable,
    env_cols: tuple = ("heat_index", "aqi"),
    shock_multipliers: tuple = (1.0, 1.25, 1.5, 2.0),
) -> pd.DataFrame:
    """Multiply environmental features by escalating shock factors.

    The point is to see whether performance gracefully degrades or
    catastrophically fails when conditions move outside the empirical range.
    """
    rows = []
    y = df[y_col].to_numpy()
    base_X = df[feature_cols].to_numpy()
    base_cols = list(feature_cols)
    for mult in shock_multipliers:
        Xp = base_X.copy()
        for c in env_cols:
            if c in base_cols:
                idx = base_cols.index(c)
                Xp[:, idx] = Xp[:, idx] * mult
        y_pred = predict_fn(Xp)
        m = classification_metrics(y, y_pred)
        m["shock_multiplier"] = float(mult)
        rows.append(m)
    return pd.DataFrame(rows)
