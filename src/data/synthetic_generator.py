"""Synthetic multimodal cohort generator.

This module produces a 500-participant × 180-day synthetic dataset with
plausible climate-physiology coupling, behavioral rhythms, delayed
physiological effects, and state-dependent missingness. It is the default
data source used throughout the repository so that the full pipeline
(features, encoder, dynamics, perturbations, dashboard) can run without
any private data.

The generative process is deliberately hand-specified rather than
GAN-fit, so the causal structure is inspectable. Each participant has a
small static profile (age, sex, chronotype, baselines, climate
vulnerability, resilience) and their daily streams evolve under a
specified set of couplings.

Couplings encoded (all approximate, none clinically validated):

* heat raises nighttime resting HR and degrades sleep efficiency
* poor air quality lowers sleep efficiency and raises fatigue
* a poor sleep night lowers next-day HRV (1-day lag)
* low HRV raises stress load
* stress load raises missingness probability and degrades mood
* screen time delays sleep midpoint and shortens sleep duration
* activity raises recovery with diminishing returns
* climate-vulnerable participants react more strongly to environmental shocks
* heat waves act as forcing shocks with carryover
* weekday/weekend rhythms in screen, mobility, and sleep midpoint

Latent ground-truth states are emitted alongside observations so that
representation analyses can be validated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.paths import SYNTHETIC_DIR, ensure_dir

# ---------------------------------------------------------------------------
# Static participant profile
# ---------------------------------------------------------------------------

CHRONOTYPES = ("morning", "intermediate", "evening")
SEXES = ("F", "M", "NB")


@dataclass(frozen=True)
class ParticipantProfile:
    """Per-participant static parameters."""

    participant_id: str
    age: int
    sex: str
    chronotype: str
    baseline_sleep_need: float       # hours
    baseline_hrv: float              # ms (RMSSD-like)
    baseline_resting_hr: float       # bpm
    baseline_activity_level: float   # 0..1
    baseline_screen_time: float      # minutes/day
    baseline_stress_sensitivity: float
    baseline_climate_vulnerability: float
    baseline_resilience: float
    baseline_missingness_tendency: float


def _sample_profiles(rng: np.random.Generator, n: int) -> list[ParticipantProfile]:
    profiles: list[ParticipantProfile] = []
    for i in range(n):
        age = int(np.clip(rng.normal(34, 12), 18, 75))
        sex = SEXES[int(rng.integers(0, 3))]
        chrono = CHRONOTYPES[int(rng.choice(3, p=[0.25, 0.5, 0.25]))]

        # Baseline HRV declines with age, with substantial individual variation
        base_hrv = float(np.clip(60.0 - 0.4 * (age - 30) + rng.normal(0, 12), 15, 110))
        base_rhr = float(np.clip(64.0 + 0.15 * (age - 30) + rng.normal(0, 6), 45, 95))
        base_sleep = float(np.clip(rng.normal(7.6, 0.7), 5.5, 9.5))
        base_act = float(np.clip(rng.beta(2.5, 2.5), 0.05, 0.95))
        base_screen = float(np.clip(rng.normal(220, 70), 30, 600))

        stress_sens = float(np.clip(rng.beta(2, 5), 0.05, 0.95))
        climate_vuln = float(np.clip(rng.beta(2, 6), 0.02, 0.95))
        resilience = float(np.clip(rng.beta(5, 2), 0.1, 0.98))
        miss_tend = float(np.clip(rng.beta(1.5, 8), 0.01, 0.6))

        profiles.append(
            ParticipantProfile(
                participant_id=f"P{i:04d}",
                age=age,
                sex=sex,
                chronotype=chrono,
                baseline_sleep_need=base_sleep,
                baseline_hrv=base_hrv,
                baseline_resting_hr=base_rhr,
                baseline_activity_level=base_act,
                baseline_screen_time=base_screen,
                baseline_stress_sensitivity=stress_sens,
                baseline_climate_vulnerability=climate_vuln,
                baseline_resilience=resilience,
                baseline_missingness_tendency=miss_tend,
            )
        )
    return profiles


# ---------------------------------------------------------------------------
# Environment simulation (shared across the cohort)
# ---------------------------------------------------------------------------


def _simulate_environment(
    n_days: int,
    start_date: date,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Daily environment, simulated as a shared exposure trajectory.

    Real cohorts would have per-participant location resolution; for a
    synthetic prototype we share a single exposure series across
    participants and let individual climate vulnerability scale the
    response.
    """
    day_of_year = np.array(
        [(start_date + timedelta(days=int(t))).timetuple().tm_yday for t in range(n_days)]
    )

    # Seasonal sinusoid (Northern-hemisphere phase: peak ~ day 200) +
    # autoregressive weather noise.
    seasonal = 20.0 + 12.0 * np.cos(2 * np.pi * (day_of_year - 200) / 365.0)
    noise = np.zeros(n_days)
    for t in range(1, n_days):
        noise[t] = 0.75 * noise[t - 1] + rng.normal(0, 2.4)
    temperature_c = seasonal + noise

    # Nighttime ~ daytime - 7°C + jitter
    nighttime_temperature_c = temperature_c - 7.0 + rng.normal(0, 1.5, n_days)

    humidity = np.clip(55 + 12 * np.cos(2 * np.pi * (day_of_year - 220) / 365.0)
                       + rng.normal(0, 8, n_days), 18, 98)

    # AQI as positively-skewed log-normal with episodic spikes
    aqi = np.clip(np.exp(rng.normal(3.7, 0.45, n_days)), 8, 380)
    n_spikes = max(2, int(n_days * 0.04))
    for idx in rng.integers(0, n_days, size=n_spikes):
        decay = np.exp(-np.arange(7) / 1.6)
        end = min(idx + 7, n_days)
        aqi[idx:end] = np.maximum(aqi[idx:end], (220 + rng.normal(0, 25)) * decay[: end - idx])
    aqi = np.clip(aqi, 8, 500)

    heat_index = temperature_c + 0.07 * np.maximum(humidity - 50, 0)

    # Heat wave: 3+ consecutive days with heat_index > 32
    heat_wave_flag = np.zeros(n_days, dtype=int)
    run = 0
    for t in range(n_days):
        if heat_index[t] > 32:
            run += 1
        else:
            run = 0
        if run >= 3:
            heat_wave_flag[max(0, t - run + 1) : t + 1] = 1

    poor_air_quality_flag = (aqi > 150).astype(int)

    dates = [start_date + timedelta(days=int(t)) for t in range(n_days)]
    return pd.DataFrame(
        {
            "date": dates,
            "temperature_c": temperature_c.round(2),
            "nighttime_temperature_c": nighttime_temperature_c.round(2),
            "humidity": humidity.round(1),
            "aqi": aqi.round(1),
            "heat_index": heat_index.round(2),
            "heat_wave_flag": heat_wave_flag,
            "poor_air_quality_flag": poor_air_quality_flag,
        }
    )


# ---------------------------------------------------------------------------
# Per-participant daily simulation
# ---------------------------------------------------------------------------


def _logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _chronotype_midpoint_target(chrono: str) -> float:
    """Preferred sleep midpoint in hours (0–24 clock; 3.5 = 03:30)."""
    return {"morning": 2.5, "intermediate": 3.5, "evening": 4.8}[chrono]


def _simulate_participant(
    profile: ParticipantProfile,
    env: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a participant's day-by-day record."""
    n = len(env)
    dates = env["date"].to_list()
    temp = env["temperature_c"].to_numpy()
    night_temp = env["nighttime_temperature_c"].to_numpy()
    aqi = env["aqi"].to_numpy()
    heat_index = env["heat_index"].to_numpy()
    heat_wave = env["heat_wave_flag"].to_numpy()
    weekday = np.array([d.weekday() for d in dates])  # 0=Mon, 6=Sun
    is_weekend = (weekday >= 5).astype(float)

    # Latent states (smoothed; ground truth)
    Z_recovery = np.zeros(n)
    Z_circadian = np.zeros(n)
    Z_stress = np.zeros(n)
    Z_env = np.zeros(n)
    Z_behav = np.zeros(n)
    Z_miss = np.zeros(n)

    # Observed streams
    sleep_duration = np.zeros(n)
    sleep_efficiency = np.zeros(n)
    sleep_midpoint = np.zeros(n)
    hrv_rmssd = np.zeros(n)
    resting_hr = np.zeros(n)
    daily_steps = np.zeros(n)
    active_minutes = np.zeros(n)
    recovery_score = np.zeros(n)
    stress_score = np.zeros(n)
    screen_time = np.zeros(n)
    phone_unlocks = np.zeros(n)
    mobility_radius = np.zeros(n)
    location_entropy = np.zeros(n)
    behavioral_regularity = np.zeros(n)
    mood = np.zeros(n)
    fatigue = np.zeros(n)
    perceived_stress = np.zeros(n)
    energy = np.zeros(n)
    cognitive_load_proxy = np.zeros(n)

    target_midpoint = _chronotype_midpoint_target(profile.chronotype)
    vuln = profile.baseline_climate_vulnerability
    sens = profile.baseline_stress_sensitivity
    resil = profile.baseline_resilience
    half_life = 1.0 + 6.0 * resil  # days
    decay = np.exp(-np.log(2) / half_life)

    # Carryover env burden
    env_burden_carry = 0.0

    for t in range(n):
        # ---- Environmental burden (forcing) ---------------------------
        epl_raw = (
            0.30 * (heat_index[t] / 40.0)
            + 0.30 * np.clip((night_temp[t] - 18) / 12.0, -0.5, 1.5)
            + 0.25 * np.clip(aqi[t] / 200.0, 0, 2.5)
            + 0.15 * heat_wave[t]
        )
        epl = vuln * epl_raw
        env_burden_carry = 0.6 * env_burden_carry + 0.4 * epl
        Z_env[t] = env_burden_carry + rng.normal(0, 0.03)

        # ---- Screen and behavior --------------------------------------
        weekend_screen = 1.0 + 0.18 * is_weekend[t]
        prior_stress = Z_stress[t - 1] if t > 0 else 0.0
        screen_t = profile.baseline_screen_time * weekend_screen * (1.0 + 0.25 * prior_stress)
        screen_time[t] = float(np.clip(screen_t + rng.normal(0, 25), 10, 720))

        phone_unlocks[t] = float(
            np.clip(60 + 80 * profile.baseline_screen_time / 240
                    + 20 * is_weekend[t] + 30 * prior_stress + rng.normal(0, 15), 5, 400)
        )

        # ---- Sleep midpoint (circadian) ------------------------------
        midpoint_delay_from_screen = 0.0025 * (screen_time[t] - 180)  # hours
        midpoint_delay_from_weekend = 0.55 * is_weekend[t]
        sleep_midpoint[t] = target_midpoint + midpoint_delay_from_screen \
            + midpoint_delay_from_weekend + rng.normal(0, 0.3)

        # ---- Sleep duration / efficiency ------------------------------
        screen_penalty = max(0.0, (screen_time[t] - 180) / 600)
        heat_penalty = max(0.0, (night_temp[t] - 22) * vuln * 0.06)
        aqi_penalty = max(0.0, (aqi[t] - 80) / 800) * vuln
        sleep_duration[t] = float(np.clip(
            profile.baseline_sleep_need - 0.7 * screen_penalty - heat_penalty
            - 0.4 * prior_stress * sens + rng.normal(0, 0.3),
            3.5, 11.0,
        ))
        sleep_efficiency[t] = float(np.clip(
            0.92 - 0.05 * screen_penalty - 0.06 * (heat_penalty / max(heat_penalty + 0.1, 1.0))
            - 0.05 * aqi_penalty - 0.05 * prior_stress * sens + rng.normal(0, 0.03),
            0.5, 0.99,
        ))

        # ---- HRV (with 1-day lag from sleep) --------------------------
        if t == 0:
            prior_sleep_duration = profile.baseline_sleep_need
            prior_sleep_eff = 0.88
        else:
            prior_sleep_duration = sleep_duration[t - 1]
            prior_sleep_eff = sleep_efficiency[t - 1]
        sleep_quality_proxy = (prior_sleep_duration / profile.baseline_sleep_need) \
            * (prior_sleep_eff / 0.88)
        hrv_t = profile.baseline_hrv * (0.6 + 0.4 * sleep_quality_proxy) \
            - 4.0 * Z_env[t] - 4.0 * prior_stress * sens + rng.normal(0, 3.5)
        hrv_rmssd[t] = float(np.clip(hrv_t, 6.0, 160.0))

        # ---- Resting HR -----------------------------------------------
        rhr_t = profile.baseline_resting_hr + 3.5 * Z_env[t] + 2.0 * prior_stress * sens \
            - 1.5 * (sleep_quality_proxy - 1.0) + rng.normal(0, 1.5)
        resting_hr[t] = float(np.clip(rhr_t, 40.0, 110.0))

        # ---- Activity --------------------------------------------------
        act_base = profile.baseline_activity_level * (0.8 + 0.2 * is_weekend[t])
        activity_t = act_base * (1.0 - 0.3 * Z_env[t]) * (1.0 - 0.2 * prior_stress * sens)
        active_minutes[t] = float(np.clip(activity_t * 90 + rng.normal(0, 12), 0, 240))
        daily_steps[t] = float(np.clip(activity_t * 12000 + rng.normal(0, 1400), 200, 30000))

        # ---- Latent autonomic recovery --------------------------------
        recovery_raw = (
            0.5 * (hrv_rmssd[t] / max(profile.baseline_hrv, 1.0) - 1.0)
            - 0.3 * (resting_hr[t] / max(profile.baseline_resting_hr, 1.0) - 1.0)
            + 0.2 * (sleep_quality_proxy - 1.0)
        )
        Z_recovery[t] = 0.7 * (Z_recovery[t - 1] if t > 0 else 0.0) + 0.3 * recovery_raw \
            + rng.normal(0, 0.03)

        recovery_score[t] = float(np.clip(60 + 25 * Z_recovery[t] + rng.normal(0, 3), 5, 100))

        # ---- Latent stress load (carryover) ---------------------------
        stress_raw = (
            0.4 * Z_env[t]
            - 0.5 * Z_recovery[t]
            + 0.2 * screen_penalty
            + 0.2 * max(0.0, (sleep_midpoint[t] - target_midpoint))
        ) * (0.7 + sens)
        Z_stress[t] = decay * (Z_stress[t - 1] if t > 0 else 0.0) + (1 - decay) * stress_raw \
            + rng.normal(0, 0.03)
        stress_score[t] = float(np.clip(40 + 30 * Z_stress[t] + rng.normal(0, 4), 0, 100))

        # ---- Circadian alignment (latent) -----------------------------
        if t >= 7:
            mid_var = float(np.var(sleep_midpoint[t - 6 : t + 1]))
        else:
            mid_var = float(np.var(sleep_midpoint[: t + 1])) if t > 0 else 0.1
        align_raw = -mid_var - 0.3 * abs(sleep_midpoint[t] - target_midpoint)
        Z_circadian[t] = 0.7 * (Z_circadian[t - 1] if t > 0 else 0.0) + 0.3 * align_raw \
            + rng.normal(0, 0.03)

        # ---- Behavior: mobility, regularity ---------------------------
        mobility_radius[t] = float(np.clip(
            (8.0 if not is_weekend[t] else 14.0)
            * (1.0 - 0.3 * Z_stress[t])
            + rng.normal(0, 3.0), 0.2, 60.0,
        ))
        location_entropy[t] = float(np.clip(
            0.6 + 0.3 * is_weekend[t] - 0.2 * Z_stress[t] + rng.normal(0, 0.08), 0.05, 1.5,
        ))
        behavioral_regularity[t] = float(np.clip(
            0.8 - 0.4 * Z_stress[t] - 0.2 * abs(sleep_midpoint[t] - target_midpoint)
            + rng.normal(0, 0.06), 0.0, 1.0,
        ))

        # Behavioral instability latent
        Z_behav[t] = 0.7 * (Z_behav[t - 1] if t > 0 else 0.0) \
            + 0.3 * (-behavioral_regularity[t] + 0.5) + rng.normal(0, 0.03)

        # ---- Surveys / proxies ----------------------------------------
        mood[t] = float(np.clip(60 - 20 * Z_stress[t] + 12 * Z_recovery[t]
                                - 8 * Z_env[t] + rng.normal(0, 4), 0, 100))
        fatigue[t] = float(np.clip(40 + 25 * Z_stress[t] - 15 * Z_recovery[t]
                                   + 10 * Z_env[t] + 6 * aqi_penalty + rng.normal(0, 4), 0, 100))
        perceived_stress[t] = float(np.clip(35 + 30 * Z_stress[t] + 8 * Z_env[t]
                                            + rng.normal(0, 4), 0, 100))
        energy[t] = float(np.clip(55 - 15 * Z_stress[t] + 18 * Z_recovery[t]
                                  + rng.normal(0, 4), 0, 100))
        cognitive_load_proxy[t] = float(np.clip(
            40 + 20 * Z_stress[t] + 12 * (screen_time[t] / 240) - 8 * Z_recovery[t]
            + rng.normal(0, 3), 0, 100,
        ))

    # ----- Missingness (state-dependent) ----------------------------------
    Z_miss = (
        0.4 * Z_stress
        + 0.3 * Z_behav
        + 0.2 * np.maximum(Z_env, 0)
        + 0.5 * profile.baseline_missingness_tendency
    )
    p_wear = _logistic(-2.5 + 1.6 * Z_miss)
    p_phone = _logistic(-2.8 + 1.4 * Z_miss)
    p_survey = _logistic(-1.9 + 2.0 * Z_miss)
    missing_wearable_flag = (rng.random(n) < p_wear).astype(int)
    missing_phone_flag = (rng.random(n) < p_phone).astype(int)
    missing_survey_flag = (rng.random(n) < p_survey).astype(int)
    total_missing_modalities = missing_wearable_flag + missing_phone_flag + missing_survey_flag

    # Mask observations where the corresponding modality is missing
    def _mask(arr: np.ndarray, flag: np.ndarray) -> np.ndarray:
        out = arr.copy().astype(float)
        out[flag == 1] = np.nan
        return out

    hrv_rmssd = _mask(hrv_rmssd, missing_wearable_flag)
    resting_hr = _mask(resting_hr, missing_wearable_flag)
    daily_steps = _mask(daily_steps, missing_wearable_flag)
    active_minutes = _mask(active_minutes, missing_wearable_flag)
    recovery_score = _mask(recovery_score, missing_wearable_flag)
    stress_score = _mask(stress_score, missing_wearable_flag)
    screen_time = _mask(screen_time, missing_phone_flag)
    phone_unlocks = _mask(phone_unlocks, missing_phone_flag)
    mobility_radius = _mask(mobility_radius, missing_phone_flag)
    location_entropy = _mask(location_entropy, missing_phone_flag)
    mood = _mask(mood, missing_survey_flag)
    fatigue = _mask(fatigue, missing_survey_flag)
    perceived_stress = _mask(perceived_stress, missing_survey_flag)
    energy = _mask(energy, missing_survey_flag)
    cognitive_load_proxy = _mask(cognitive_load_proxy, missing_survey_flag)

    # ----- Regime label (assigned from latent state thresholds) -----------
    regime = _assign_regime(Z_stress, Z_recovery, Z_env)

    df = pd.DataFrame(
        {
            "participant_id": profile.participant_id,
            "date": dates,
            # static
            "age": profile.age,
            "sex": profile.sex,
            "chronotype": profile.chronotype,
            "baseline_sleep_need": profile.baseline_sleep_need,
            "baseline_hrv": profile.baseline_hrv,
            "baseline_resting_hr": profile.baseline_resting_hr,
            "baseline_activity_level": profile.baseline_activity_level,
            "baseline_screen_time": profile.baseline_screen_time,
            "baseline_stress_sensitivity": profile.baseline_stress_sensitivity,
            "baseline_climate_vulnerability": profile.baseline_climate_vulnerability,
            "baseline_resilience": profile.baseline_resilience,
            "baseline_missingness_tendency": profile.baseline_missingness_tendency,
            # daily physiology
            "sleep_duration": sleep_duration,
            "sleep_efficiency": sleep_efficiency,
            "sleep_midpoint": sleep_midpoint,
            "hrv_rmssd": hrv_rmssd,
            "resting_hr": resting_hr,
            "daily_steps": daily_steps,
            "active_minutes": active_minutes,
            "recovery_score": recovery_score,
            "stress_score": stress_score,
            # behavior
            "screen_time_minutes": screen_time,
            "phone_unlock_count": phone_unlocks,
            "mobility_radius_km": mobility_radius,
            "location_entropy": location_entropy,
            "behavioral_regularity": behavioral_regularity,
            # environment
            "temperature_c": temp,
            "nighttime_temperature_c": night_temp,
            "humidity": env["humidity"].to_numpy(),
            "aqi": aqi,
            "heat_index": heat_index,
            "heat_wave_flag": heat_wave,
            "poor_air_quality_flag": env["poor_air_quality_flag"].to_numpy(),
            # surveys
            "mood_score": mood,
            "fatigue_score": fatigue,
            "perceived_stress": perceived_stress,
            "energy_score": energy,
            "cognitive_load_proxy": cognitive_load_proxy,
            # latent ground truth
            "latent_autonomic_recovery": Z_recovery,
            "latent_circadian_alignment": Z_circadian,
            "latent_stress_load": Z_stress,
            "latent_environmental_burden": Z_env,
            "latent_behavioral_instability": Z_behav,
            "latent_missingness_pressure": Z_miss,
            # regime
            "regime_label": regime,
            # missingness
            "missing_wearable_flag": missing_wearable_flag,
            "missing_phone_flag": missing_phone_flag,
            "missing_survey_flag": missing_survey_flag,
            "total_missing_modalities": total_missing_modalities,
        }
    )
    return df


def _assign_regime(Z_stress: np.ndarray, Z_recovery: np.ndarray, Z_env: np.ndarray) -> np.ndarray:
    """Assign one of {stable, stressed, dysregulated, recovery} per day."""
    n = len(Z_stress)
    out = np.empty(n, dtype=object)
    # Thresholds chosen empirically against the synthetic latent ranges so
    # that all four regimes are populated in the default cohort.
    for t in range(n):
        s, r, e = Z_stress[t], Z_recovery[t], Z_env[t]
        ds = Z_stress[t] - Z_stress[t - 1] if t > 0 else 0.0
        if s > 0.30 and r < -0.05:
            out[t] = "dysregulated"
        elif s > 0.12 or e > 0.12:
            out[t] = "stressed"
        elif r > 0.05 and ds < -0.01:
            out[t] = "recovery"
        else:
            out[t] = "stable"
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_synthetic_cohort(
    n_participants: int = 500,
    n_days: int = 180,
    start_date: str | date = "2024-01-01",
    seed: int = 17,
) -> pd.DataFrame:
    """Generate the full synthetic cohort as a single long-format DataFrame.

    Parameters
    ----------
    n_participants : int
        Number of synthetic participants.
    n_days : int
        Number of daily observations per participant.
    start_date : str or date
        Cohort start date.
    seed : int
        RNG seed; controls all stochastic components.

    Returns
    -------
    pandas.DataFrame
        Long-format frame with one row per participant-day.
    """
    rng = np.random.default_rng(seed)
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    env = _simulate_environment(n_days, start_date, rng)
    profiles = _sample_profiles(rng, n_participants)

    frames = [_simulate_participant(p, env, rng) for p in profiles]
    cohort = pd.concat(frames, axis=0, ignore_index=True)
    cohort = cohort.sort_values(["participant_id", "date"]).reset_index(drop=True)
    # Normalise the date column to pandas datetime64 so callers can use
    # the .dt accessor regardless of whether the cohort came straight from
    # the generator or via a CSV round-trip.
    cohort["date"] = pd.to_datetime(cohort["date"])
    return cohort


def save_synthetic_data(
    df: pd.DataFrame,
    path: Optional[str | Path] = None,
) -> Path:
    """Write the cohort to CSV. Returns the resolved path."""
    target = Path(path) if path else (SYNTHETIC_DIR / "synthetic_cohort.csv")
    ensure_dir(target.parent)
    df.to_csv(target, index=False)
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic multimodal cohort.")
    p.add_argument("--participants", type=int, default=500)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--start-date", type=str, default="2024-01-01")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> Path:
    args = _build_arg_parser().parse_args(argv)
    df = generate_synthetic_cohort(
        n_participants=args.participants,
        n_days=args.days,
        start_date=args.start_date,
        seed=args.seed,
    )
    out = save_synthetic_data(df, path=args.out)
    print(f"[synthetic_generator] wrote {len(df):,} rows to {out}")
    return out


if __name__ == "__main__":
    main()
