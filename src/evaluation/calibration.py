"""Calibration diagnostics."""

from __future__ import annotations

import numpy as np


def calibration_curve_data(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> dict:
    """Empirical reliability curve for a binary probabilistic prediction.

    Returns a dict with `prob_pred`, `prob_true`, and `bin_count` arrays of
    length `n_bins`, suitable for plotting.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_proba, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    prob_pred = np.zeros(n_bins)
    prob_true = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            prob_pred[b] = y_proba[m].mean()
            prob_true[b] = y_true[m].mean()
            counts[b] = m.sum()
    return {"prob_pred": prob_pred, "prob_true": prob_true, "bin_count": counts}


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: weighted absolute deviation between predicted and empirical probability."""
    d = calibration_curve_data(y_true, y_proba, n_bins=n_bins)
    total = d["bin_count"].sum() or 1.0
    weights = d["bin_count"] / total
    return float(np.sum(weights * np.abs(d["prob_pred"] - d["prob_true"])))
