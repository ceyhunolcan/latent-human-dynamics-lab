"""Multi-step intervention scenarios.

Wraps the perturbation engine in a slightly higher-level scenario object.
Useful for the dashboard and the API, where the caller wants to specify a
scenario by name and get back a structured report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .perturbation_engine import (
    PerturbationSpec,
    PerturbationResult,
    simulate_perturbation,
    available_perturbations,
)
from dynamics.transition_model import LatentDynamicsModel


@dataclass
class InterventionScenario:
    participant_id: str
    perturbation_type: str
    magnitude: Optional[float] = None
    horizon_days: int = 14
    resilience_half_life_days: float = 4.0
    vulnerability_coefficient: float = 1.0
    notes: str = ""


def run_intervention_scenario(
    scenario: InterventionScenario,
    Z_recent: np.ndarray,
    environment_forecast: np.ndarray,
    behavior_forecast: np.ndarray,
    dynamics_model: Optional[LatentDynamicsModel] = None,
) -> PerturbationResult:
    """Run a named scenario and return a `PerturbationResult`."""
    if scenario.perturbation_type not in available_perturbations():
        raise ValueError(
            f"perturbation type {scenario.perturbation_type!r} not in "
            f"{available_perturbations()}"
        )
    spec = PerturbationSpec(
        perturbation_type=scenario.perturbation_type,
        magnitude=scenario.magnitude,
        horizon_days=scenario.horizon_days,
        resilience_half_life_days=scenario.resilience_half_life_days,
        vulnerability_coefficient=scenario.vulnerability_coefficient,
    )
    return simulate_perturbation(
        Z_recent=Z_recent,
        environment_forecast=environment_forecast,
        behavior_forecast=behavior_forecast,
        spec=spec,
        dynamics_model=dynamics_model,
    )


def summarise_intervention_pathway(result: PerturbationResult) -> str:
    """Produce a short human-readable summary of an intervention result.

    The summary lists the largest proxy shifts and the pathway explanation,
    framed as a research simulation. Always carries the disclaimer.
    """
    proxies = result.proxy_delta_mean
    if not proxies:
        body = "Insufficient information to summarise proxy-level shifts."
    else:
        items = sorted(proxies.items(), key=lambda kv: -abs(kv[1]))[:5]
        body_lines = []
        for k, v in items:
            sign = "+" if v >= 0 else ""
            body_lines.append(f"  {k}: {sign}{v:.3f}")
        body = "Largest projected proxy shifts (research simulation):\n" + "\n".join(body_lines)

    return (
        f"Scenario: {result.spec.perturbation_type} "
        f"(magnitude {result.spec.resolved_magnitude()}, horizon {result.spec.horizon_days} days)\n\n"
        f"{body}\n\n"
        f"Pathway: {result.pathway_explanation}\n\n"
        f"{result.disclaimer}"
    )
