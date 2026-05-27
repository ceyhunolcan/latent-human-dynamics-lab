"""Latent state space layer.

This package implements the core scientific contribution: representing each
participant as a trajectory in a six-dimensional latent space whose axes are
interpretable (autonomic recovery, circadian alignment, stress load,
environmental burden, behavioral instability, missingness pressure), and
analysing that trajectory for regime structure, energy basins, and
critical-transition warning signals.

Nothing here is clinical. The latent states are research constructs intended
to support hypothesis generation and counterfactual reasoning under explicit
modelling assumptions.
"""

from .latent_state_encoder import (
    MultimodalLatentStateEncoder,
    encode_latent_states_classical,
    reconstruction_loss,
    next_state_prediction_loss,
    smoothness_regularization,
    contrastive_trajectory_loss,
    disentanglement_penalty,
)
from .regime_detector import (
    RegimeDetector,
    fit_regime_detector,
    transition_matrix,
    regime_stability_score,
)
from .state_geometry import project_pca, project_latent_2d, trajectory_curvature
from .energy_landscape import (
    estimate_energy_landscape,
    plot_energy_landscape,
    sample_energy_at_points,
)
from .early_warning import (
    rolling_variance_signal,
    rolling_autocorrelation_signal,
    instability_index,
    distance_to_dysregulated_centroid,
    critical_transition_warning_score,
)

__all__ = [
    "MultimodalLatentStateEncoder",
    "encode_latent_states_classical",
    "reconstruction_loss",
    "next_state_prediction_loss",
    "smoothness_regularization",
    "contrastive_trajectory_loss",
    "disentanglement_penalty",
    "RegimeDetector",
    "fit_regime_detector",
    "transition_matrix",
    "regime_stability_score",
    "project_pca",
    "project_latent_2d",
    "trajectory_curvature",
    "estimate_energy_landscape",
    "plot_energy_landscape",
    "sample_energy_at_points",
    "rolling_variance_signal",
    "rolling_autocorrelation_signal",
    "instability_index",
    "distance_to_dysregulated_centroid",
    "critical_transition_warning_score",
]
