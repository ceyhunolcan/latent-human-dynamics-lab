"""Latent state geometry.

Tools for projecting the 6-dimensional latent space to 2D for visualisation,
and for summarising the geometry of individual participant trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAProjection:
    components: np.ndarray  # (k, d)
    mean: np.ndarray  # (d,)
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components.T


def project_pca(Z: np.ndarray, n_components: int = 2) -> PCAProjection:
    """Fit a PCA on Z and return a small projection object."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=17)
    pca.fit(Z)
    return PCAProjection(
        components=pca.components_,
        mean=pca.mean_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def project_latent_2d(Z: np.ndarray) -> np.ndarray:
    """Convenience: return a (T, 2) PCA projection for visualisation."""
    proj = project_pca(Z, n_components=2)
    return proj.transform(Z)


def trajectory_curvature(Z_traj: np.ndarray) -> np.ndarray:
    """Discrete curvature along a trajectory.

    Returns one curvature value per interior point. Useful for spotting
    sharp pivots in latent state that may correspond to regime shifts.
    """
    if Z_traj.shape[0] < 3:
        return np.zeros(max(Z_traj.shape[0] - 2, 0))
    v1 = Z_traj[1:-1] - Z_traj[:-2]
    v2 = Z_traj[2:] - Z_traj[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-9
    n2 = np.linalg.norm(v2, axis=1) + 1e-9
    cos_theta = (v1 * v2).sum(axis=1) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
