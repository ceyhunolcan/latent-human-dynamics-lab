"""Perturbation engine.

Public API for simulating counterfactual perturbations on a participant's
recent latent trajectory. Each call returns a `PerturbationResult` with:

* baseline trajectory in latent space
* counterfactual trajectory in latent space
* per-dimension latent-state delta (mean + uncertainty)
* implied observed proxy delta (sleep duration, HRV, fatigue, ...)
* uncertainty bands
* a plain-language pathway explanation
* the standard non-clinical disclaimer

The seven canonical perturbations from the spec are supported:
sleep_extension, screen_reduction, exercise_boost, cooling,
air_quality_improvement, heat_wave_shock, combined_resilience_protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynamics.forcing_functions import apply_perturbation_operator
from dynamics.transition_model import LatentDynamicsModel
from safety.output_disclaimer import DISCLAIMER, attach_non_clinical_warning
from safety.clinical_guardrails import validate_safe_output
from safety.risk_language import convert_risk_language_to_research_language
from .trajectory_comparator import compare_trajectories, latent_to_observed_proxies


# Configured magnitudes per spec / configs/perturbations.yaml defaults.
_DEFAULT_MAGNITUDES = {
    "sleep_extension": 45.0,            # +minutes of sleep
    "screen_reduction": 60.0,           # -minutes of screen time
    "exercise_boost": 30.0,             # +active minutes
    "cooling": -4.0,                    # delta nighttime temp in °C
    "air_quality_improvement": -40.0,   # delta AQI
    "heat_wave_shock": 6.0,             # +°C heat
    "combined_resilience_protocol": 1.0,
}

_PATHWAY_EXPLANATIONS = {
    "sleep_extension": (
        "Extending sleep duration is expected to raise autonomic recovery and "
        "improve circadian alignment, with a downstream reduction in stress load. "
        "The effect decays over several days unless the change is sustained."
    ),
    "screen_reduction": (
        "Reducing late-evening screen exposure is expected to improve circadian "
        "alignment and behavioural regularity, indirectly supporting recovery."
    ),
    "exercise_boost": (
        "Additional active minutes are expected to support autonomic recovery "
        "and modestly reduce accumulated stress, with diminishing returns at "
        "high baseline activity."
    ),
    "cooling": (
        "Lowering nighttime ambient temperature is expected to reduce environmental "
        "burden, improve sleep continuity, and indirectly raise HRV. The effect "
        "is larger for climate-sensitive participants."
    ),
    "air_quality_improvement": (
        "Reducing AQI is expected to lower environmental burden and stress load "
        "and slightly improve recovery, with a smaller effect on circadian "
        "alignment."
    ),
    "heat_wave_shock": (
        "Sustained heat-wave exposure is expected to suppress autonomic recovery, "
        "destabilise circadian alignment, and raise stress and missingness "
        "pressure. The effect compounds over consecutive days."
    ),
    "combined_resilience_protocol": (
        "A combined protocol of sleep extension, additional activity, and "
        "nighttime cooling is expected to move the trajectory toward the "
        "stable regime across multiple latent dimensions simultaneously."
    ),
}


def available_perturbations() -> list:
    """Return the list of supported perturbation type names."""
    return list(_DEFAULT_MAGNITUDES.keys())


def load_perturbation_defaults_from_config(config: Optional[dict] = None) -> dict:
    """Resolve perturbation default magnitudes from a loaded config dict.

    Lookup path: config["perturbations"]["perturbations"][<type>]["default"].
    Falls back to _DEFAULT_MAGNITUDES for any type without a YAML default.
    """
    if config is None:
        try:
            from utils.config import load_default_config
            config = load_default_config()
        except Exception:
            return dict(_DEFAULT_MAGNITUDES)

    magnitudes = dict(_DEFAULT_MAGNITUDES)
    try:
        yaml_perts = config.get("perturbations", {}).get("perturbations", {})
        for ptype, spec in yaml_perts.items():
            if not isinstance(spec, dict):
                continue
            default = spec.get("default")
            if isinstance(default, (int, float)):
                if ptype in magnitudes:
                    magnitudes[ptype] = float(default)
    except (AttributeError, TypeError):
        pass
    return magnitudes


@dataclass
class PerturbationSpec:
    perturbation_type: str
    magnitude: Optional[float] = None
    horizon_days: int = 14
    resilience_half_life_days: float = 4.0
    vulnerability_coefficient: float = 1.0

    def resolved_magnitude(self) -> float:
        if self.magnitude is not None:
            m = float(self.magnitude)
            if not np.isfinite(m):
                raise ValueError(
                    f"perturbation magnitude must be finite; got {self.magnitude!r}. "
                    "Pass None to use the default for this perturbation type."
                )
            return m
        if self.perturbation_type not in _DEFAULT_MAGNITUDES:
            raise ValueError(
                f"unknown perturbation type: {self.perturbation_type!r}. "
                f"Valid types: {sorted(_DEFAULT_MAGNITUDES.keys())}"
            )
        return _DEFAULT_MAGNITUDES[self.perturbation_type]


@dataclass
class PerturbationResult:
    spec: PerturbationSpec
    baseline_latent: np.ndarray
    counterfactual_latent: np.ndarray
    latent_delta_mean: np.ndarray  # (horizon, latent_dim)
    proxy_delta_mean: dict
    proxy_delta_std: dict
    pathway_explanation: str
    disclaimer: str = DISCLAIMER

    # ------------------------------------------------------------------
    # Aliases for downstream consumers (API, dashboard, scripts, tests).
    # The canonical fields end in `_latent`; the aliases use the more
    # conventional `_trajectory` and `observed_*` names that consumers expect.
    # ------------------------------------------------------------------
    @property
    def baseline_trajectory(self) -> np.ndarray:
        return self.baseline_latent

    @property
    def counterfactual_trajectory(self) -> np.ndarray:
        return self.counterfactual_latent

    @property
    def observed_proxy_delta(self) -> dict:
        return self.proxy_delta_mean

    @property
    def latent_state_delta(self) -> np.ndarray:
        # Per-dim mean latent change across the horizon
        return np.asarray(self.latent_delta_mean).mean(axis=0)

    @property
    def magnitude(self) -> float:
        return self.spec.resolved_magnitude()

    @property
    def horizon_days(self) -> int:
        return self.spec.horizon_days

    def to_dict(self) -> dict:
        base = np.asarray(self.baseline_latent)
        cf = np.asarray(self.counterfactual_latent)
        d = {
            "perturbation_type": self.spec.perturbation_type,
            "magnitude": self.spec.resolved_magnitude(),
            "horizon_days": self.spec.horizon_days,
            "baseline_trajectory": base.tolist(),
            "counterfactual_trajectory": cf.tolist(),
            "latent_state_delta": self.latent_state_delta.tolist(),
            "latent_delta_mean_per_dim": {
                f"dim_{i}": float(self.latent_delta_mean[:, i].mean())
                for i in range(self.latent_delta_mean.shape[1])
            },
            "observed_proxy_delta": self.proxy_delta_mean,
            "proxy_delta_mean": self.proxy_delta_mean,
            "proxy_delta_std": self.proxy_delta_std,
            "uncertainty": (
                {"proxy_std": self.proxy_delta_std}
                if self.proxy_delta_std
                else None
            ),
            "pathway_explanation": self.pathway_explanation,
            "disclaimer": self.disclaimer,
        }
        return d


def simulate_perturbation(
    Z_recent: np.ndarray,
    environment_forecast: np.ndarray,
    behavior_forecast: np.ndarray,
    spec: PerturbationSpec,
    dynamics_model: Optional[LatentDynamicsModel] = None,
    uncertainty_std: Optional[np.ndarray] = None,
) -> PerturbationResult:
    """Simulate baseline vs counterfactual trajectories under a perturbation.

    Parameters
    ----------
    Z_recent
        Recent observed/inferred latent states for the participant; shape
        (n_recent, latent_dim). The most recent row is used as the initial
        condition for the forecast.
    environment_forecast, behavior_forecast
        Forecasted forcing inputs over the perturbation horizon. Shape
        (horizon_days, 4) each.
    spec
        The perturbation specification.
    dynamics_model
        Optional pre-configured dynamics model. A default is constructed if
        absent.
    uncertainty_std
        Optional per-step per-dimension std for the counterfactual; defaults
        to a global value of 0.02 across all dims.
    """
    if dynamics_model is None:
        dynamics_model = LatentDynamicsModel()

    Z_recent = np.asarray(Z_recent, dtype=float)
    if Z_recent.ndim == 1:
        Z0 = Z_recent
    else:
        Z0 = Z_recent[-1]
    horizon = spec.horizon_days

    # Align the forecast windows to `horizon`. If the caller provided a
    # longer forecast we truncate; if shorter, we repeat the last observed
    # row. This guarantees the baseline trajectory has length `horizon`
    # and matches the perturbation operator's output shape.
    env = np.asarray(environment_forecast, dtype=float)
    beh = np.asarray(behavior_forecast, dtype=float)
    if env.shape[0] >= horizon:
        env = env[:horizon]
    else:
        env = np.concatenate(
            [env, np.tile(env[-1:], (horizon - env.shape[0], 1))]
            if env.shape[0] > 0
            else [np.zeros((horizon, 4))],
            axis=0,
        )
    if beh.shape[0] >= horizon:
        beh = beh[:horizon]
    else:
        beh = np.concatenate(
            [beh, np.tile(beh[-1:], (horizon - beh.shape[0], 1))]
            if beh.shape[0] > 0
            else [np.zeros((horizon, 4))],
            axis=0,
        )

    baseline_traj = dynamics_model.simulate_trajectory(
        Z_0=Z0,
        environment=env,
        behavior=beh,
        vulnerability_coefficient=spec.vulnerability_coefficient,
    )[1:]  # drop initial condition for delta computation

    perturbation_delta = apply_perturbation_operator(
        Z=Z0,
        perturbation_type=spec.perturbation_type,
        magnitude=spec.resolved_magnitude(),
        horizon_days=horizon,
        resilience_half_life_days=spec.resilience_half_life_days,
        vulnerability_coefficient=spec.vulnerability_coefficient,
    )

    # We simulate the counterfactual by adding the perturbation operator on
    # top of the same baseline trajectory. This is equivalent to assuming
    # the perturbation is small enough that the dynamics are locally linear;
    # the assumption is documented in paper/mechanistic_formalism.md.
    counter_traj = baseline_traj + perturbation_delta

    if uncertainty_std is None:
        uncertainty_std = np.full_like(counter_traj, 0.02)

    comparison = compare_trajectories(
        baseline_latent=baseline_traj,
        counterfactual_latent=counter_traj,
        uncertainty_std=uncertainty_std,
    )

    explanation = _PATHWAY_EXPLANATIONS.get(
        spec.perturbation_type,
        "Pathway explanation unavailable for this perturbation type.",
    )

    result = PerturbationResult(
        spec=spec,
        baseline_latent=baseline_traj,
        counterfactual_latent=counter_traj,
        latent_delta_mean=comparison.latent_delta,
        proxy_delta_mean=comparison.proxy_delta_mean,
        proxy_delta_std=comparison.proxy_delta_std,
        pathway_explanation=explanation,
    )

    # Final safety pass: sanitise the language and ensure no clinical claims
    # slipped through the explanation strings.
    sanitized = validate_safe_output(result.to_dict(), strict=False)
    result.pathway_explanation = convert_risk_language_to_research_language(
        sanitized.get("pathway_explanation", explanation)
    )
    return result
