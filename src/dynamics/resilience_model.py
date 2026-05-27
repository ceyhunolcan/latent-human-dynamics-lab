"""Personalised resilience modelling.

Estimates per-participant resilience parameters from their observed
trajectory: how quickly stress markers return toward baseline after a
perturbation, how persistent stress is, how sensitive the participant is
to environmental forcing, and how unstable their behavioural rhythms are.

The five canonical profiles below are convenience labels, not clinical
categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


RESILIENCE_PROFILES = (
    "high_resilience",
    "moderate_resilience",
    "climate_vulnerable",
    "recovery_delayed",
    "behaviorally_unstable",
)


@dataclass
class ResilienceProfile:
    participant_id: str
    recovery_half_life_days: float
    stress_persistence: float
    environmental_sensitivity: float
    behavioral_instability_index: float
    profile_label: str

    def to_dict(self) -> dict:
        return {
            "participant_id": self.participant_id,
            "recovery_half_life_days": self.recovery_half_life_days,
            "stress_persistence": self.stress_persistence,
            "environmental_sensitivity": self.environmental_sensitivity,
            "behavioral_instability_index": self.behavioral_instability_index,
            "profile_label": self.profile_label,
        }


def recovery_half_life(stress_series: np.ndarray, min_obs: int = 14) -> float:
    """Estimate the half-life of a stress signal by fitting log(stress) ~ -lambda * t.

    Returns the half-life in days. If the fit fails or the signal is too
    short, returns a default of 4 days, matching the global prior.
    """
    s = np.asarray(stress_series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < min_obs:
        return 4.0

    # Only shift if the signal contains non-positive values. Shifting a
    # positive exponential by its minimum DESTROYS the exponential structure
    # because log(A e^(-λt) - A_min + ε) is not linear in t. For clean
    # decay signals we want to log directly.
    if s.min() > 0:
        y = np.log(s)
    else:
        # Mixed-sign or non-positive signal: shift just enough to make it
        # positive. This loses some precision but preserves orderability.
        y = np.log(s - s.min() + 1.0)
    t = np.arange(len(y))
    try:
        slope, _ = np.polyfit(t, y, 1)
    except Exception:
        return 4.0
    if slope >= 0:
        return 14.0  # no recovery detectable
    return float(min(max(np.log(2) / (-slope), 0.5), 21.0))


def apply_resilience_decay(
    perturbation: np.ndarray,
    half_life_days: float,
    n_steps: int,
) -> np.ndarray:
    """Decay an immediate perturbation vector geometrically over n_steps days."""
    t = np.arange(n_steps)
    decay = np.exp(-np.log(2) * t / max(half_life_days, 0.5))
    return decay[:, None] * np.asarray(perturbation)[None, :]


def estimate_resilience_profile(participant_df: pd.DataFrame, participant_id: str) -> ResilienceProfile:
    """Estimate a profile from a single participant's daily trajectory.

    Inputs are robust to missing columns: if a signal is unavailable we fall
    back to the global prior for that quantity.
    """
    df = participant_df.copy()

    # Recovery half-life from stress_score if available, else from
    # latent_stress_load, else default.
    if "stress_score" in df.columns and df["stress_score"].notna().sum() >= 14:
        half_life = recovery_half_life(df["stress_score"].to_numpy())
    elif "latent_stress_load" in df.columns:
        half_life = recovery_half_life(df["latent_stress_load"].to_numpy())
    else:
        half_life = 4.0

    # Stress persistence: lag-1 autocorrelation of stress.
    if "stress_score" in df.columns:
        s = df["stress_score"].dropna().to_numpy()
        if len(s) > 14:
            stress_persistence = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        else:
            stress_persistence = 0.4
    else:
        stress_persistence = 0.4

    # Environmental sensitivity: correlation between heat_index and recovery.
    if "heat_index" in df.columns and "recovery_score" in df.columns:
        try:
            env_sens = -float(df["heat_index"].corr(df["recovery_score"]))
        except Exception:
            env_sens = 0.3
    else:
        env_sens = 0.3
    if np.isnan(env_sens):
        env_sens = 0.3

    # Behavioural instability: rolling std of sleep midpoint.
    if "sleep_midpoint" in df.columns:
        bi = float(df["sleep_midpoint"].std())
    else:
        bi = 1.0

    label = _label_profile(half_life, stress_persistence, env_sens, bi)
    return ResilienceProfile(
        participant_id=str(participant_id),
        recovery_half_life_days=half_life,
        stress_persistence=stress_persistence,
        environmental_sensitivity=env_sens,
        behavioral_instability_index=bi,
        profile_label=label,
    )


def _label_profile(half_life: float, persistence: float, env_sens: float, bi: float) -> str:
    if env_sens > 0.5:
        return "climate_vulnerable"
    if half_life > 8:
        return "recovery_delayed"
    if bi > 1.5:
        return "behaviorally_unstable"
    if half_life < 3.0 and persistence < 0.3:
        return "high_resilience"
    return "moderate_resilience"
