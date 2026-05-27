"""Latent transition model.

This is the public entry point to the dynamics layer. The model wraps a
single-step transition function (GRU or neural ODE) together with the
forcing functions defined in `forcing_functions.py`. It exposes:

* `predict_next_state(Z_t, behavior_t, environment_t, perturbation_t)`
* `simulate_trajectory(Z_0, behavior, environment, perturbation, horizon_days)`

Both work without PyTorch (a deterministic linear-Gaussian transition is
used as the fallback step) and gain a learned neural step when torch is
installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .forcing_functions import (
    environmental_forcing,
    behavioral_forcing,
    apply_perturbation_operator,
    environmental_physiological_load,
)

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class GRUTransitionStep(nn.Module):
        """One-step transition: Z_{t+1} = Z_t + GRUCell(forcing, Z_t)."""

        def __init__(self, latent_dim: int = 6, forcing_dim: int = 12, hidden_dim: int = 32):
            super().__init__()
            self.latent_dim = latent_dim
            self.forcing_dim = forcing_dim
            self.input_proj = nn.Linear(forcing_dim, hidden_dim)
            self.cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
            self.delta_head = nn.Linear(hidden_dim, latent_dim)
            self.hidden = None

        def forward(self, z, u):
            h_in = self.input_proj(u)
            if self.hidden is None or self.hidden.shape[0] != z.shape[0]:
                self.hidden = torch.zeros(z.shape[0], self.input_proj.out_features, device=z.device)
            self.hidden = self.cell(h_in, self.hidden)
            return self.delta_head(self.hidden)

        def reset(self):
            self.hidden = None

else:  # pragma: no cover

    class GRUTransitionStep:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("GRUTransitionStep requires PyTorch.")


@dataclass
class LatentDynamicsModel:
    """Discrete-time latent dynamics model.

    Parameters
    ----------
    latent_dim
        Dimensionality of Z. Default 6.
    contraction
        Per-step contraction toward zero in the fallback transition. A value
        of 0.1 means the latent state relaxes 10% toward equilibrium each
        day in the absence of forcing. Used only when no learned step is
        provided.
    noise_std
        Standard deviation of the residual epsilon_t (per-dimension).
    rng_seed
        Seed for the stochastic residual sampler.
    """

    latent_dim: int = 6
    contraction: float = 0.1
    noise_std: float = 0.01
    rng_seed: int = 17
    learned_step: Optional[object] = None
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, rng_seed: int = 17):
        """Construct a `LatentDynamicsModel` from the YAML config stack.

        Falls back to the canonical paper defaults if the config file is
        missing or has no relevant keys. This lets users tweak dynamics
        constants by editing ``configs/dynamics.yaml`` without touching code.
        """
        from utils.config import get_dynamics_settings
        s = get_dynamics_settings(cfg)
        return cls(
            contraction=s["contraction"],
            noise_std=s["noise_std"],
            rng_seed=rng_seed,
        )

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_next_state(
        self,
        Z_t: np.ndarray,
        behavior_t: np.ndarray,
        environment_t: np.ndarray,
        perturbation_t: Optional[np.ndarray] = None,
        vulnerability_coefficient: float = 1.0,
    ) -> np.ndarray:
        """Single-step prediction.

        Inputs are 1D arrays for one participant at one time step.
        `environment_t` is expected to have 4 columns: [daytime_heat,
        nighttime_heat, aqi, heatwave_exposure_days].
        `behavior_t` is expected to have 4 columns: [sleep_z, activity_z,
        screen_z, social_rhythm_z].
        """
        if not np.isfinite(vulnerability_coefficient):
            raise ValueError(
                f"vulnerability_coefficient must be finite; got "
                f"{vulnerability_coefficient!r}. Use 1.0 for default."
            )
        Z = np.atleast_1d(Z_t).astype(float)
        env = np.atleast_1d(environment_t).astype(float)
        beh = np.atleast_1d(behavior_t).astype(float)

        # Latent state itself must be finite — there is no sensible "neutral"
        # value to impute, and if Z is NaN it means the encoder upstream
        # already failed. Surface this loudly.
        if not np.isfinite(Z).all():
            raise ValueError(
                f"Z_t contains non-finite values: {Z}. "
                "Latent state must be finite; check upstream encoder/imputation."
            )

        # NaN in env/behavior inputs is normal for real cohorts (missing days,
        # sensors off, etc.). Impute with the normalization-reference values
        # so that after z-scoring inside the forcing functions, the signal
        # contribution is approximately zero (interpreted as "neutral day
        # for this place / person", which is the right epistemic stance when
        # we have no data). Without this, NaN propagates silently to Z_next.
        _ENV_REF = np.array([18.0, 14.0, 50.0, 0.0])  # matches forcing_functions normalization
        _BEH_REF = np.array([0.0, 0.0, 0.0, 0.0])      # behavior inputs already z-scored typically
        env = np.where(np.isfinite(env), env, _ENV_REF[: len(env)])
        beh = np.where(np.isfinite(beh), beh, _BEH_REF[: len(beh)])

        # Forcings. The dynamics layer enforces input normalization so
        # callers can pass raw cohort values (temperature_c, screen_minutes)
        # without producing runaway trajectories.
        F_env = environmental_forcing(
            env[0:1], env[1:2], env[2:3], env[3:4],
            vulnerability_coefficient=vulnerability_coefficient,
            latent_dim=self.latent_dim,
            normalize_inputs=True,
        )[0]
        F_beh = behavioral_forcing(
            beh[0:1], beh[1:2], beh[2:3], beh[3:4],
            latent_dim=self.latent_dim,
            normalize_inputs=True,
        )[0]

        # Drift toward zero plus forcing.
        drift = -self.contraction * Z + 0.04 * F_env + 0.04 * F_beh
        Z_next = Z + drift

        if perturbation_t is not None:
            Z_next = Z_next + np.asarray(perturbation_t, dtype=float)

        # Additive Gaussian residual.
        Z_next = Z_next + self._rng.normal(0.0, self.noise_std, size=self.latent_dim)
        return Z_next

    # ------------------------------------------------------------------
    # Trajectory simulation
    # ------------------------------------------------------------------

    def simulate_trajectory(
        self,
        Z_0: np.ndarray,
        environment: np.ndarray,
        behavior: np.ndarray,
        perturbation: Optional[np.ndarray] = None,
        vulnerability_coefficient: float = 1.0,
    ) -> np.ndarray:
        """Simulate forward from Z_0 over the length of environment.

        `environment` shape: (T, 4). `behavior` shape: (T, 4). `perturbation`
        if provided is shape (T, latent_dim) of additive corrections to Z.

        Returns trajectory of shape (T + 1, latent_dim).
        """
        env = np.asarray(environment, dtype=float)
        beh = np.asarray(behavior, dtype=float)
        T = env.shape[0]
        Z = np.asarray(Z_0, dtype=float).copy()
        traj = [Z.copy()]
        for t in range(T):
            p = perturbation[t] if perturbation is not None else None
            Z = self.predict_next_state(Z, beh[t], env[t], perturbation_t=p,
                                        vulnerability_coefficient=vulnerability_coefficient)
            traj.append(Z.copy())
        return np.stack(traj, axis=0)
