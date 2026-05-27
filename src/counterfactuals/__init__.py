"""Counterfactual perturbation engine.

Given a participant's recent latent trajectory and a candidate perturbation
(sleep extension, cooling, AQI improvement, etc.), this layer simulates what
would have happened *with* and *without* the perturbation and reports the
difference in latent state and observed proxies, together with an
uncertainty band and a non-clinical explanation.

All outputs are wrapped through the safety layer before they leave the
package, so every counterfactual reply carries the standard research-only
disclaimer.
"""

from .perturbation_engine import (
    PerturbationSpec,
    PerturbationResult,
    simulate_perturbation,
    available_perturbations,
)
from .intervention_simulator import (
    InterventionScenario,
    run_intervention_scenario,
    summarise_intervention_pathway,
)
from .trajectory_comparator import (
    TrajectoryComparison,
    compare_trajectories,
    latent_to_observed_proxies,
)

__all__ = [
    "PerturbationSpec",
    "PerturbationResult",
    "simulate_perturbation",
    "available_perturbations",
    "InterventionScenario",
    "run_intervention_scenario",
    "summarise_intervention_pathway",
    "TrajectoryComparison",
    "compare_trajectories",
    "latent_to_observed_proxies",
]
