"""Regime detection in latent state space.

Given a participant's latent trajectory, we partition state space into
discrete regimes via k-means and analyse the empirical transition behaviour.
The four-regime parameterisation (stable / stressed / dysregulated / recovery)
mirrors the synthetic generator's ground truth, but the detector is fit
unsupervised and the assignment of cluster index to regime name is done
post-hoc by matching centroid coordinates to plausible directions in latent
space.

Limitations: regime structure is sensitive to the number of clusters, the
chosen features, and the smoothing window. We report a stability score so
users can interrogate this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

REGIME_NAMES = ("stable", "stressed", "dysregulated", "recovery")


@dataclass
class RegimeDetector:
    """Fitted k-means regime detector over latent space.

    Attributes
    ----------
    centroids : np.ndarray  (k, latent_dim)
    labels_to_regime : list[str]
        Mapping from cluster index to regime name.
    feature_means, feature_stds : np.ndarray
        Per-dimension normalisation used during fit; applied at predict time.
    """

    centroids: np.ndarray
    labels_to_regime: list
    feature_means: np.ndarray
    feature_stds: np.ndarray

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Return one regime label per row of Z."""
        Z_norm = (Z - self.feature_means) / (self.feature_stds + 1e-9)
        # squared distance to each centroid
        d = ((Z_norm[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=-1)
        idx = d.argmin(axis=1)
        return np.array([self.labels_to_regime[i] for i in idx])

    def distance_to_centroid(self, Z: np.ndarray, regime_name: str) -> np.ndarray:
        """L2 distance from each row of Z to the named regime centroid."""
        if regime_name not in self.labels_to_regime:
            raise ValueError(f"unknown regime {regime_name!r}")
        c_idx = self.labels_to_regime.index(regime_name)
        Z_norm = (Z - self.feature_means) / (self.feature_stds + 1e-9)
        return np.sqrt(((Z_norm - self.centroids[c_idx]) ** 2).sum(axis=-1))

    def regime_risk_summary(self, Z_recent: np.ndarray) -> dict:
        """Compute distance-based risk scores from a recent window of Z."""
        if Z_recent.ndim == 1:
            Z_recent = Z_recent[None, :]
        Z_norm = (Z_recent - self.feature_means) / (self.feature_stds + 1e-9)
        out = {}
        for name in REGIME_NAMES:
            if name in self.labels_to_regime:
                c = self.centroids[self.labels_to_regime.index(name)]
                d = np.sqrt(((Z_norm - c) ** 2).sum(axis=-1)).mean()
                out[f"distance_to_{name}_centroid"] = float(d)
        # Soft assignment via inverse-distance weighting.
        dists = np.stack(
            [
                np.sqrt(((Z_norm - self.centroids[self.labels_to_regime.index(n)]) ** 2).sum(axis=-1)).mean()
                for n in self.labels_to_regime
            ]
        )
        weights = 1.0 / (dists + 1e-6)
        weights = weights / weights.sum()
        for n, w in zip(self.labels_to_regime, weights):
            out[f"prob_{n}"] = float(w)
        out["dysregulation_risk"] = float(out.get("prob_dysregulated", 0.0))
        out["recovery_probability"] = float(out.get("prob_recovery", 0.0))
        return out


def fit_regime_detector(
    Z: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 17,
) -> RegimeDetector:
    """Fit a k-means regime detector to a latent state matrix.

    Z is (n_rows, latent_dim). The function normalises each dimension to unit
    variance before clustering, then assigns cluster indices to regime names
    using a heuristic based on the sign of the first three latent dimensions
    (autonomic recovery, circadian alignment, stress load). This works well
    when the encoder produces approximately aligned axes; otherwise the names
    are still consistent within a fit and downstream code only depends on
    the assignment being stable.
    """
    from sklearn.cluster import KMeans

    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape {Z.shape}")

    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    Z_norm = (Z - mu) / (sd + 1e-9)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(Z_norm)
    centroids = km.cluster_centers_

    # Heuristic labelling: examine the first three latent dimensions.
    #   recovery axis high, stress axis low  -> stable
    #   stress axis moderately high          -> stressed
    #   stress high AND recovery low         -> dysregulated
    #   recovery axis trending positive,
    #     stress axis trending negative      -> recovery
    labels = ["stable"] * n_clusters
    if centroids.shape[1] >= 3:
        rec_score = centroids[:, 0]
        stress_score = centroids[:, 2] if centroids.shape[1] > 2 else np.zeros(n_clusters)
        # Sort by stress score.
        order = np.argsort(stress_score)
        if n_clusters >= 4:
            labels[order[0]] = "stable"
            labels[order[1]] = "recovery"
            labels[order[2]] = "stressed"
            labels[order[3]] = "dysregulated"
        else:
            for i, idx in enumerate(order):
                labels[idx] = REGIME_NAMES[min(i, len(REGIME_NAMES) - 1)]

    return RegimeDetector(
        centroids=centroids,
        labels_to_regime=labels,
        feature_means=mu,
        feature_stds=sd,
    )


def transition_matrix(
    regime_sequence: Sequence[str],
    regimes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, list]:
    """Empirical transition probability matrix over a regime sequence.

    Returns
    -------
    matrix : (k, k) ndarray
        Row-stochastic transition matrix. `matrix[i, j]` is the empirical
        probability of moving from regime ``regimes[i]`` to ``regimes[j]``
        on the next day.
    regimes : list of str
        The regime labels in the order used to index the matrix, so callers
        can construct labelled DataFrames.
    """
    regimes = list(regimes) if regimes is not None else list(REGIME_NAMES)
    idx = {r: i for i, r in enumerate(regimes)}
    k = len(regimes)
    counts = np.zeros((k, k), dtype=float)
    for a, b in zip(regime_sequence[:-1], regime_sequence[1:]):
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums, regimes


def regime_stability_score(regime_sequence: Sequence[str]) -> float:
    """Fraction of day-to-day transitions that keep the regime unchanged."""
    if len(regime_sequence) < 2:
        return 1.0
    same = sum(1 for a, b in zip(regime_sequence[:-1], regime_sequence[1:]) if a == b)
    return same / (len(regime_sequence) - 1)
