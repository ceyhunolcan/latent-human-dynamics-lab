"""Representation analysis for the latent state space.

Asks: do the learned latent dimensions correspond to anything interpretable?
We answer with three diagnostics. (1) PCA of the latent space to look for
low-rank structure. (2) Silhouette-style cluster separation against the
ground-truth regime labels in the synthetic generator. (3) Correlation
between each latent dimension and each ground-truth latent axis, where
available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def representation_pca(Z: np.ndarray, n_components: int = 6) -> dict:
    """Run PCA on Z and return components plus explained variance ratio."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(n_components, Z.shape[1]), random_state=17).fit(Z)
    return {
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "singular_values": pca.singular_values_,
    }


def cluster_separation_score(Z: np.ndarray, labels) -> float:
    """Silhouette score of `labels` over `Z`. Higher is better; range [-1, 1]."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(labels) < 5:
        return float("nan")
    try:
        return float(silhouette_score(Z, labels))
    except Exception:
        return float("nan")


def latent_to_ground_truth_correlation(
    Z: np.ndarray,
    ground_truth_df: pd.DataFrame,
    ground_truth_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Pearson correlation between each latent dim and each ground-truth axis.

    Returns a (latent_dim, ground_truth) dataframe.
    """
    if ground_truth_cols is None:
        ground_truth_cols = [
            c for c in ground_truth_df.columns if c.startswith("latent_")
        ]
    if not ground_truth_cols:
        return pd.DataFrame()
    G = ground_truth_df[ground_truth_cols].to_numpy()
    rows = []
    for i in range(Z.shape[1]):
        row = {}
        for j, gname in enumerate(ground_truth_cols):
            z = Z[:, i]
            g = G[:, j]
            if z.std() < 1e-9 or g.std() < 1e-9:
                row[gname] = float("nan")
            else:
                row[gname] = float(np.corrcoef(z, g)[0, 1])
        row["latent_dim"] = f"z_{i}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("latent_dim")


def trajectory_stability(participant_traj: np.ndarray) -> float:
    """Mean step size in latent space; lower means smoother trajectory."""
    if participant_traj.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(participant_traj, axis=0), axis=1)
    return float(steps.mean())
