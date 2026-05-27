"""Compare baseline and counterfactual latent trajectories.

The comparator translates latent-state deltas into observed-proxy deltas
using a fixed linear readout. The readout coefficients are fixed and
documented so that the inferred proxy shifts (sleep duration, HRV, fatigue)
can be inspected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Latent -> observed proxy coefficients. Each row corresponds to a proxy
# variable and each column to a latent dimension. Hand-specified so that
# the directions match the synthetic generator's coupling rules.
#
# Latent dims (cols): [autonomic_recovery, circadian_alignment, stress_load,
#                      environmental_burden, behavioral_instability, missingness_pressure]
_PROXY_READOUT = {
    "sleep_duration_hours":   np.array([+0.20, +0.30,  0.00, -0.10, -0.15,  0.00]),
    "hrv_rmssd":              np.array([+4.50, +0.50, -2.50, -1.50,  0.00,  0.00]),
    "resting_hr":             np.array([-2.00,  0.00, +2.50, +1.00,  0.00,  0.00]),
    "recovery_score":         np.array([+6.00, +1.00, -3.00, -2.00, -1.00,  0.00]),
    "stress_score":           np.array([-4.00, -1.00, +6.00, +2.00, +1.00,  0.00]),
    "fatigue_score":          np.array([-3.00, -1.00, +3.00, +1.50,  0.00,  0.00]),
    "mood_score":             np.array([+2.50, +1.50, -2.50, -1.00, -0.50,  0.00]),
    "missing_modalities":     np.array([ 0.00,  0.00,  0.00,  0.00,  0.10, +0.30]),
}


@dataclass
class TrajectoryComparison:
    baseline_latent: np.ndarray
    counterfactual_latent: np.ndarray
    latent_delta: np.ndarray
    proxy_delta_mean: dict = field(default_factory=dict)
    proxy_delta_std: dict = field(default_factory=dict)

    # Convenience aliases used by API / dashboard / tests.
    @property
    def observed_proxy_delta(self) -> dict:
        return self.proxy_delta_mean

    @property
    def uncertainty_std(self) -> dict:
        return self.proxy_delta_std


def latent_to_observed_proxies(latent_delta: np.ndarray) -> dict:
    """Project a (T, latent_dim) latent delta to observed proxy deltas.

    Returns a dict mapping proxy name to a (T,) array of expected shifts.
    """
    if latent_delta.ndim == 1:
        latent_delta = latent_delta[None, :]
    out = {}
    for name, coef in _PROXY_READOUT.items():
        if latent_delta.shape[1] != coef.shape[0]:
            continue
        out[name] = latent_delta @ coef
    return out


def compare_trajectories(
    baseline_latent: np.ndarray,
    counterfactual_latent: np.ndarray,
    uncertainty_std: Optional[np.ndarray] = None,
    n_mc_samples: int = 50,
    seed: int = 17,
) -> TrajectoryComparison:
    """Compare two latent trajectories and summarise the implied proxy shifts.

    `uncertainty_std` is an optional per-time per-dimension std for the
    counterfactual trajectory; when provided, the proxy-level uncertainty is
    estimated by MC sampling.
    """
    baseline = np.asarray(baseline_latent, dtype=float)
    counter = np.asarray(counterfactual_latent, dtype=float)
    if baseline.shape != counter.shape:
        raise ValueError(f"shape mismatch: baseline {baseline.shape} vs counter {counter.shape}")

    latent_delta = counter - baseline
    proxy_mean_paths = latent_to_observed_proxies(latent_delta)
    proxy_mean = {k: float(np.mean(v)) for k, v in proxy_mean_paths.items()}

    proxy_std = {}
    if uncertainty_std is not None:
        rng = np.random.default_rng(seed)
        std = np.asarray(uncertainty_std, dtype=float)
        if std.shape != counter.shape:
            std = np.broadcast_to(std, counter.shape)
        samples_per_proxy = {k: [] for k in proxy_mean.keys()}
        for _ in range(n_mc_samples):
            sample_counter = counter + rng.normal(0.0, std)
            sample_delta = sample_counter - baseline
            proxies = latent_to_observed_proxies(sample_delta)
            for k, v in proxies.items():
                samples_per_proxy[k].append(np.mean(v))
        proxy_std = {k: float(np.std(samples_per_proxy[k])) for k in samples_per_proxy}
    else:
        proxy_std = {k: 0.0 for k in proxy_mean}

    return TrajectoryComparison(
        baseline_latent=baseline,
        counterfactual_latent=counter,
        latent_delta=latent_delta,
        proxy_delta_mean=proxy_mean,
        proxy_delta_std=proxy_std,
    )
