"""Latent dynamics layer.

Implements the state-transition law

    dZ/dt = f(Z_t, E_t, B_t, P_t) + epsilon_t

where Z is the six-dimensional latent state, E is environmental forcing,
B is behavioural input, P is a perturbation operator, and epsilon is a
stochastic residual. Two concrete realisations of f are provided: a GRU
discrete-time transition model, and an explicit-Euler neural ODE.
"""

from .transition_model import LatentDynamicsModel, GRUTransitionStep
from .neural_ode import NeuralODEStep, euler_integrate, RK4_integrate
from .forcing_functions import (
    environmental_forcing,
    behavioral_forcing,
    environmental_physiological_load,
    apply_perturbation_operator,
)
from .resilience_model import (
    ResilienceProfile,
    estimate_resilience_profile,
    recovery_half_life,
    apply_resilience_decay,
)

__all__ = [
    "LatentDynamicsModel",
    "GRUTransitionStep",
    "NeuralODEStep",
    "euler_integrate",
    "RK4_integrate",
    "environmental_forcing",
    "behavioral_forcing",
    "environmental_physiological_load",
    "apply_perturbation_operator",
    "ResilienceProfile",
    "estimate_resilience_profile",
    "recovery_half_life",
    "apply_resilience_decay",
]
