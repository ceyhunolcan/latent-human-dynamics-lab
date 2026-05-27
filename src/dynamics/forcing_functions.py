"""Forcing functions for latent dynamics.

These functions translate raw environmental and behavioural inputs into
forcing vectors that act on the latent state. They are deterministic by
construction so reviewers can audit every coefficient.

Formal definitions appear in paper/mechanistic_formalism.md.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Default weights for Environmental Physiological Load (EPL), matched to
# configs/dynamics.yaml. Keep in sync.
EPL_DEFAULT_WEIGHTS = {
    "daytime_heat": 0.30,
    "nighttime_heat": 0.30,
    "aqi": 0.25,
    "heatwave": 0.15,
}


def load_epl_weights_from_config(config: Optional[dict] = None) -> dict:
    """Resolve EPL weights from a loaded config dict, falling back to defaults.

    Lookup path: config["dynamics"]["forcing"]["epl_weights"]. If the config
    isn't passed, loads the default config via utils.config.load_default_config.

    Returns a dict with the canonical EPL keys (daytime_heat, nighttime_heat,
    aqi, heatwave). Any missing keys fall back to EPL_DEFAULT_WEIGHTS.
    """
    if config is None:
        try:
            from utils.config import load_default_config
            config = load_default_config()
        except Exception:
            return dict(EPL_DEFAULT_WEIGHTS)

    weights = dict(EPL_DEFAULT_WEIGHTS)
    try:
        yaml_weights = (
            config.get("dynamics", {}).get("forcing", {}).get("epl_weights", {})
        )
        for k, v in yaml_weights.items():
            if k in weights and isinstance(v, (int, float)):
                weights[k] = float(v)
    except (AttributeError, TypeError):
        pass
    return weights


def environmental_physiological_load(
    daytime_heat: float,
    nighttime_heat: float,
    aqi: float,
    heatwave_exposure_days: float,
    weights: Optional[dict] = None,
) -> float:
    """Compute scalar EPL from environmental components.

    Inputs should be z-scored at the cohort level for the weights to combine
    on a comparable scale. Returns a scalar.
    """
    w = weights or EPL_DEFAULT_WEIGHTS
    return (
        w["daytime_heat"] * float(daytime_heat)
        + w["nighttime_heat"] * float(nighttime_heat)
        + w["aqi"] * float(aqi)
        + w["heatwave"] * float(heatwave_exposure_days)
    )


def environmental_forcing(
    daytime_heat: np.ndarray,
    nighttime_heat: np.ndarray,
    aqi: np.ndarray,
    heatwave_exposure_days: np.ndarray,
    vulnerability_coefficient: float = 1.0,
    latent_dim: int = 6,
    normalize_inputs: bool = True,
    epl_weights: Optional[dict] = None,
) -> np.ndarray:
    """Project environmental inputs onto the latent state space.

    Returns a (T, latent_dim) array shaped to be added to dZ/dt. The mapping
    is:

        autonomic_recovery        <-  -EPL  (heat/AQI degrades recovery)
        circadian_alignment       <-  -nighttime_heat
        stress_load               <-  +EPL
        environmental_burden      <-  +EPL  (direct mapping)
        behavioral_instability    <-  +0.4 * EPL
        missingness_pressure      <-  +0.2 * heatwave

    Each row is then scaled by `vulnerability_coefficient` (>1 for climate-
    sensitive participants, <1 for resilient).

    When ``normalize_inputs=True`` (default), the four environmental
    channels are mapped from raw units into z-score-like deviations against
    rough population reference points. This makes the function safe to
    call with raw daily values from a cohort dataframe (e.g.
    temperature_c=28.0, aqi=85.0). Set ``normalize_inputs=False`` if the
    caller has already standardized the inputs.

    EPL weights default to the constants in EPL_DEFAULT_WEIGHTS. Pass a
    ``weights`` dict (or load one from configs/dynamics.yaml via
    ``load_epl_weights_from_config``) to override them.
    """
    # Reference points and scales for default normalization. These are
    # rough population priors for a temperate climate; they're chosen so
    # ordinary days produce |EPL| < 1 and extreme days produce |EPL| in
    # the 2-3 range.
    if normalize_inputs:
        daytime_heat = (np.asarray(daytime_heat, dtype=float) - 18.0) / 8.0
        nighttime_heat = (np.asarray(nighttime_heat, dtype=float) - 14.0) / 6.0
        aqi = (np.asarray(aqi, dtype=float) - 50.0) / 40.0
        heatwave_exposure_days = np.asarray(heatwave_exposure_days, dtype=float)

    T = len(daytime_heat)
    EPL = np.array(
        [
            environmental_physiological_load(
                daytime_heat[t], nighttime_heat[t], aqi[t], heatwave_exposure_days[t],
                weights=epl_weights,
            )
            for t in range(T)
        ]
    )
    F = np.zeros((T, latent_dim))
    F[:, 0] = -EPL                              # autonomic_recovery
    F[:, 1] = -np.asarray(nighttime_heat)       # circadian_alignment
    F[:, 2] = EPL                               # stress_load
    F[:, 3] = EPL                               # environmental_burden
    F[:, 4] = 0.4 * EPL                         # behavioral_instability
    F[:, 5] = 0.2 * np.asarray(heatwave_exposure_days)  # missingness_pressure
    return F * float(vulnerability_coefficient)


def behavioral_forcing(
    sleep_z: np.ndarray,
    activity_z: np.ndarray,
    screen_z: np.ndarray,
    social_rhythm_z: np.ndarray,
    latent_dim: int = 6,
    normalize_inputs: bool = False,
) -> np.ndarray:
    """Project behavioural inputs onto the latent state space.

    Mapping:
        autonomic_recovery        <-  +sleep + 0.3 * activity
        circadian_alignment       <-  +sleep - 0.4 * screen + 0.3 * social_rhythm
        stress_load               <-  -0.3 * sleep + 0.2 * screen
        environmental_burden      <-  0
        behavioral_instability    <-  -social_rhythm + 0.3 * screen
        missingness_pressure      <-  -0.2 * activity

    Inputs are expected to be z-scored at the cohort level. If
    ``normalize_inputs=True``, a permissive squash is applied so raw values
    don't blow up downstream dynamics. The default (False) preserves the
    documented contract for callers that have done their own z-scoring.
    """
    sleep = np.asarray(sleep_z, dtype=float)
    act = np.asarray(activity_z, dtype=float)
    scr = np.asarray(screen_z, dtype=float)
    soc = np.asarray(social_rhythm_z, dtype=float)

    if normalize_inputs:
        # Soft tanh squash to keep arbitrary-scale inputs bounded
        sleep = np.tanh(sleep / 10.0)
        act = np.tanh(act / 10.0)
        scr = np.tanh(scr / 10.0)
        soc = np.tanh(soc / 10.0)

    T = len(sleep)
    F = np.zeros((T, latent_dim))
    F[:, 0] = sleep + 0.3 * act
    F[:, 1] = sleep - 0.4 * scr + 0.3 * soc
    F[:, 2] = -0.3 * sleep + 0.2 * scr
    F[:, 3] = 0.0
    F[:, 4] = -soc + 0.3 * scr
    F[:, 5] = -0.2 * act
    return F


def apply_perturbation_operator(
    Z: np.ndarray,
    perturbation_type: str,
    magnitude: float,
    horizon_days: int,
    resilience_half_life_days: float = 4.0,
    vulnerability_coefficient: float = 1.0,
) -> np.ndarray:
    """Apply a closed-form perturbation operator over a horizon.

    Returns the *additive* trajectory perturbation Delta_Z of shape
    (horizon_days, latent_dim) that should be added to the baseline forecast
    to obtain the counterfactual forecast.

    The shape of the response per perturbation type is hard-coded against
    the directions in `paper/mechanistic_formalism.md`. The magnitude scales
    the immediate response, and the perturbation decays geometrically with
    the participant's resilience half-life.
    """
    latent_dim = Z.shape[-1] if Z.ndim > 1 else 6
    delta_immediate = np.zeros(latent_dim)

    pt = perturbation_type.lower()
    m = float(magnitude)
    v = float(vulnerability_coefficient)

    if pt == "sleep_extension":
        # +sleep improves recovery, alignment, reduces stress
        delta_immediate[0] = +0.04 * m * 0.5
        delta_immediate[1] = +0.03 * m * 0.5
        delta_immediate[2] = -0.02 * m * 0.5
    elif pt == "screen_reduction":
        # -screen improves circadian alignment and behavioral stability
        delta_immediate[1] = +0.025 * m
        delta_immediate[4] = -0.015 * m
    elif pt == "exercise_boost":
        # +activity improves recovery, modestly stress relief
        delta_immediate[0] = +0.025 * m
        delta_immediate[2] = -0.01 * m
    elif pt == "cooling":
        # -nighttime temp: reduce env burden, improve alignment & recovery
        delta_immediate[0] = +0.05 * abs(m)
        delta_immediate[1] = +0.04 * abs(m)
        delta_immediate[3] = -0.08 * abs(m)
    elif pt == "air_quality_improvement":
        # -AQI: reduce env burden and stress, improve recovery
        delta_immediate[0] = +0.03 * abs(m) * 0.05
        delta_immediate[2] = -0.02 * abs(m) * 0.05
        delta_immediate[3] = -0.05 * abs(m) * 0.05
    elif pt == "heat_wave_shock":
        # +heat: worsen everything for climate-vulnerable participants
        delta_immediate[0] = -0.06 * m * v
        delta_immediate[1] = -0.04 * m * v
        delta_immediate[2] = +0.05 * m * v
        delta_immediate[3] = +0.09 * m * v
        delta_immediate[5] = +0.02 * m * v
    elif pt == "combined_resilience_protocol":
        # Combined protocol: sleep extension + cooling + activity.
        # Magnitude is dimensionless (1.0 = "standard protocol"), and
        # for visual comparability with single-perturbation magnitudes
        # we scale the immediate delta by `m`.
        delta_immediate[0] = +0.07 * m
        delta_immediate[1] = +0.06 * m
        delta_immediate[2] = -0.04 * m
        delta_immediate[3] = -0.04 * m
        delta_immediate[4] = -0.02 * m
    else:
        valid = sorted(_PERTURBATION_DIRECTIONS.keys()) if "_PERTURBATION_DIRECTIONS" in dir() else [
            "sleep_extension", "screen_reduction", "exercise_boost", "cooling",
            "air_quality_improvement", "heat_wave_shock", "combined_resilience_protocol",
        ]
        raise ValueError(
            f"unknown perturbation type: {perturbation_type!r}. "
            f"Valid types: {valid}"
        )

    # Decay over the horizon. Negative perturbations decay toward zero;
    # positive ones do too. The half-life controls how quickly the effect
    # fades (longer = more persistent benefit/harm).
    t = np.arange(horizon_days)
    decay = np.exp(-np.log(2) * t / max(resilience_half_life_days, 0.5))
    # Heat wave shocks ramp up rather than decay; flip the curve.
    if pt == "heat_wave_shock":
        decay = 1.0 - decay * 0.5
    delta = decay[:, None] * delta_immediate[None, :]
    return delta
