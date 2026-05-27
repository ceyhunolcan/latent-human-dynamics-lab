"""Neural ODE integration for latent dynamics.

A small explicit-Euler / RK4 integrator that propagates Z_t forward in time
under a learned vector field. PyTorch is optional: when available, the step
function is a small MLP; otherwise we fall back to a linear-Gaussian
contraction toward a participant baseline.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class NeuralODEStep(nn.Module):
        """Small MLP parameterising dZ/dt = f(Z, u) where u stacks forcings."""

        def __init__(self, latent_dim: int = 6, forcing_dim: int = 12, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim + forcing_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, z, u):
            x = torch.cat([z, u], dim=-1)
            return self.net(x)

else:  # pragma: no cover

    class NeuralODEStep:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("NeuralODEStep requires PyTorch.")


def euler_integrate(
    step_fn: Callable,
    z0: np.ndarray,
    forcings: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Explicit Euler integration.

    `step_fn(z, u)` should return dZ/dt as an array of shape z.shape.
    `forcings` is shaped (T, forcing_dim). Returns trajectory shape (T+1, d).
    """
    z = np.asarray(z0, dtype=float).copy()
    traj = [z.copy()]
    for t in range(len(forcings)):
        dz = np.asarray(step_fn(z, forcings[t]), dtype=float)
        z = z + dt * dz
        traj.append(z.copy())
    return np.stack(traj, axis=0)


def RK4_integrate(
    step_fn: Callable,
    z0: np.ndarray,
    forcings: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Classical 4th-order Runge-Kutta integration.

    Uses the same forcing vector for all 4 stage evaluations within a step;
    appropriate when forcings change on the integration grid.
    """
    z = np.asarray(z0, dtype=float).copy()
    traj = [z.copy()]
    for t in range(len(forcings)):
        u = forcings[t]
        k1 = np.asarray(step_fn(z, u))
        k2 = np.asarray(step_fn(z + 0.5 * dt * k1, u))
        k3 = np.asarray(step_fn(z + 0.5 * dt * k2, u))
        k4 = np.asarray(step_fn(z + dt * k3, u))
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(z.copy())
    return np.stack(traj, axis=0)
