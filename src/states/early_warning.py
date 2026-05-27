"""Critical transition detection in latent psychophysiological state space.

Implements the classical early-warning indicators from dynamical systems
theory (rising variance, rising lag-1 autocorrelation) adapted to the
latent state trajectory inferred from passive sensing.

This is exploratory analysis. The literature on critical slowing down has
documented these signals in ecological and climate systems, and a smaller
body of work has examined them in physiology and mood. Whether they
generalise to passive-sensing-derived state trajectories at the individual
level is an open empirical question.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def rolling_variance_signal(x: np.ndarray, window: int = 14) -> np.ndarray:
    """Rolling variance of a 1D signal, computed forward in time.

    The first `window-1` entries are returned as NaN.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    for i in range(window - 1, len(x)):
        out[i] = np.var(x[i - window + 1 : i + 1], ddof=1) if window > 1 else 0.0
    return out


def rolling_autocorrelation_signal(x: np.ndarray, window: int = 14, lag: int = 1) -> np.ndarray:
    """Rolling lag-`lag` autocorrelation, computed forward in time."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < window + lag:
        return out
    for i in range(window - 1, len(x)):
        seg = x[i - window + 1 : i + 1]
        a = seg[:-lag]
        b = seg[lag:]
        if a.std() < 1e-9 or b.std() < 1e-9:
            out[i] = np.nan
        else:
            out[i] = float(np.corrcoef(a, b)[0, 1])
    return out


def instability_index(Z_traj: np.ndarray, window: int = 14) -> np.ndarray:
    """Composite instability index per row of Z_traj.

    Averages the rolling variance signal across latent dimensions and adds
    the absolute rolling lag-1 autocorrelation. The combination is intended
    to be more robust than either alone.
    """
    Z = np.asarray(Z_traj, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    var_signals = np.stack([rolling_variance_signal(Z[:, j], window) for j in range(Z.shape[1])], axis=1)
    acf_signals = np.stack(
        [rolling_autocorrelation_signal(Z[:, j], window) for j in range(Z.shape[1])], axis=1
    )
    var_mean = np.nanmean(var_signals, axis=1)
    acf_mean = np.nanmean(np.abs(acf_signals), axis=1)
    return np.where(np.isnan(var_mean), 0.0, var_mean) + np.where(np.isnan(acf_mean), 0.0, acf_mean)


def distance_to_dysregulated_centroid(Z_traj: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """L2 distance from each row of Z_traj to the dysregulated regime centroid."""
    Z = np.asarray(Z_traj, dtype=float)
    return np.sqrt(((Z - centroid[None, :]) ** 2).sum(axis=-1))


def critical_transition_warning_score(
    Z_traj: np.ndarray,
    dysregulated_centroid: Optional[np.ndarray] = None,
    window: int = 14,
) -> dict:
    """Compute a composite warning score and its components.

    Returns a dict containing arrays for variance signal, autocorrelation
    signal, instability index, optional distance to the dysregulated
    centroid, and a normalised warning score in [0, 1].
    """
    Z = np.asarray(Z_traj, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]

    var_sig = np.nanmean(
        np.stack([rolling_variance_signal(Z[:, j], window) for j in range(Z.shape[1])], axis=1), axis=1
    )
    acf_sig = np.nanmean(
        np.stack(
            [np.abs(rolling_autocorrelation_signal(Z[:, j], window)) for j in range(Z.shape[1])], axis=1
        ),
        axis=1,
    )
    inst = instability_index(Z, window=window)

    components = [var_sig, acf_sig]
    if dysregulated_centroid is not None:
        dist = distance_to_dysregulated_centroid(Z, np.asarray(dysregulated_centroid, dtype=float))
        components.append(-dist)  # closer = stronger warning

    # Normalise each component robustly and combine.
    def _robust_norm(x):
        x = np.where(np.isnan(x), 0.0, x)
        lo, hi = np.percentile(x, [5, 95])
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    norm_components = [_robust_norm(c) for c in components]
    warning = np.mean(np.stack(norm_components, axis=0), axis=0)

    return {
        "variance_signal": var_sig,
        "autocorrelation_signal": acf_sig,
        "instability_index": inst,
        "distance_to_dysregulated": components[2] * -1 if dysregulated_centroid is not None else None,
        "warning_score": warning,
    }
