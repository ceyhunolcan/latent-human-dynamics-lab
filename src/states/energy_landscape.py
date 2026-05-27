"""Energy landscape over the latent state space.

Treats the empirical density of latent states as a Boltzmann-like distribution
and reports a pseudo-energy E(z) = -log p(z). Low-energy regions correspond
to attractor-like basins where participants spend most of their time; high
energy corresponds to transient or rare states.

This is a visualisation and scientific-framing device, not a clinical
inference. The choice of density estimator, bandwidth, and the projection to
2D all affect the landscape's appearance, so the function returns the grid
and density so users can audit the choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .state_geometry import project_pca, PCAProjection


@dataclass
class EnergyLandscape:
    grid_x: np.ndarray  # 1D
    grid_y: np.ndarray  # 1D
    density: np.ndarray  # 2D
    energy: np.ndarray  # 2D
    projection: PCAProjection
    method: str  # "kde" or "histogram"


def estimate_energy_landscape(
    Z: np.ndarray,
    grid_size: int = 60,
    bandwidth: Optional[float] = None,
    use_kde: bool = True,
) -> EnergyLandscape:
    """Estimate E(z) = -log p(z) over a 2D PCA projection of Z."""
    Z = np.asarray(Z, dtype=float)
    Z2, proj = _project_or_passthrough(Z)

    # Build grid.
    pad = 0.5
    xmin, xmax = Z2[:, 0].min() - pad, Z2[:, 0].max() + pad
    ymin, ymax = Z2[:, 1].min() - pad, Z2[:, 1].max() + pad
    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

    method = "histogram"
    density: np.ndarray
    if use_kde:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(Z2.T, bw_method=bandwidth) if bandwidth else gaussian_kde(Z2.T)
            density = kde(grid_pts.T).reshape(grid_size, grid_size)
            method = "kde"
        except Exception:  # pragma: no cover - scipy is in our requirements
            density = _histogram_density(Z2, xs, ys)
    else:
        density = _histogram_density(Z2, xs, ys)

    density = np.maximum(density, 1e-9)
    energy = -np.log(density)
    energy = energy - energy.min()
    return EnergyLandscape(
        grid_x=xs,
        grid_y=ys,
        density=density,
        energy=energy,
        projection=proj,
        method=method,
    )


def _project_or_passthrough(Z: np.ndarray):
    if Z.shape[1] == 2:
        return Z, PCAProjection(
            components=np.eye(2),
            mean=Z.mean(axis=0),
            explained_variance_ratio=np.array([0.5, 0.5]),
        )
    proj = project_pca(Z, n_components=2)
    return proj.transform(Z), proj


def _histogram_density(Z2: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    H, _, _ = np.histogram2d(
        Z2[:, 0], Z2[:, 1],
        bins=[xs, ys],
    )
    # Pad to match grid_size x grid_size.
    H = np.pad(H, ((0, 1), (0, 1)), mode="edge")
    H = H / max(H.sum(), 1.0)
    return H.T  # transpose so axes match meshgrid convention


def sample_energy_at_points(landscape: EnergyLandscape, points_2d: np.ndarray) -> np.ndarray:
    """Look up energy at arbitrary 2D points by nearest-neighbour grid."""
    out = np.zeros(len(points_2d))
    for i, (x, y) in enumerate(points_2d):
        ix = int(np.clip(np.searchsorted(landscape.grid_x, x), 0, len(landscape.grid_x) - 1))
        iy = int(np.clip(np.searchsorted(landscape.grid_y, y), 0, len(landscape.grid_y) - 1))
        out[i] = landscape.energy[iy, ix]
    return out


def plot_energy_landscape(landscape: EnergyLandscape, ax=None, title: str = "Latent energy landscape"):
    """Render a contour plot of the energy landscape.

    Returns the matplotlib Axes. Caller is responsible for showing/saving.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    XX, YY = np.meshgrid(landscape.grid_x, landscape.grid_y)
    cs = ax.contourf(XX, YY, landscape.energy, levels=15, cmap="viridis")
    ax.set_xlabel("Latent PC1")
    ax.set_ylabel("Latent PC2")
    ax.set_title(title)
    plt.colorbar(cs, ax=ax, label="E(z) = -log p(z)")
    return ax
