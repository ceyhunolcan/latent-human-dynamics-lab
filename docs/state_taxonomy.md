# Latent state taxonomy

This document defines the six coordinates of the latent state vector $Z_t \in \mathbb{R}^6$ that the repository uses throughout. Each coordinate is named, has a fixed sign convention, is grounded in a handful of observed proxies, and carries a stated interpretive limit. None of these constructs are clinical diagnoses or validated biomarkers. They are research-defined coordinates that let the dynamics, perturbation, and regime-detection machinery operate on a stable, low-dimensional representation.

The taxonomy is fixed by design. Adding a coordinate is a deliberate modeling decision, not a hyperparameter sweep.

## 1. `autonomic_recovery`

**Construct.** Capacity for parasympathetic and cardiovascular recovery between days of physiological load. Higher values correspond to faster recovery and larger vagal reserve.

**Direction.** Higher = better.

**Observed proxies.**

- `hrv_rmssd` (positive)
- `resting_hr` (negative)
- `recovery_score` (positive)
- one-day-lag deviation of HRV from personal baseline (positive)

**Expected forcing.** Decreases under heat exposure (especially nighttime), under poor sleep, and under accumulated stress load.

**Interpretive limits.** A high score reflects the participant's HRV-based signal relative to their own baseline, not absolute cardiovascular fitness. It cannot be compared across participants without explicit normalization.

## 2. `circadian_alignment`

**Construct.** Stability and phase appropriateness of the sleep-wake schedule relative to the participant's chronotype. Higher values correspond to more regular sleep midpoints and a smaller phase delay relative to the preferred window.

**Direction.** Higher = more aligned.

**Observed proxies.**

- variance of `sleep_midpoint` over a 7-day rolling window (negative)
- deviation of `sleep_midpoint` from chronotype-implied target (negative)
- `sleep_duration` regularity (positive)
- weekday/weekend midpoint gap (negative)

**Expected forcing.** Decreases under high evening screen exposure, late-night activity, and disrupted weekly rhythms.

**Interpretive limits.** This is not a phase estimate in melatonin/DLMO terms. It is a behavioral regularity index that correlates with circadian alignment in the population sense.

## 3. `stress_load`

**Construct.** Accumulated psychophysiological stress burden. A short-memory exponentially-weighted aggregate of acute stress signals.

**Direction.** Higher = worse.

**Observed proxies.**

- `stress_score`
- `perceived_stress`
- negative HRV deviation
- mood-fatigue composite

**Expected forcing.** Rises under environmental burden, poor sleep, low autonomic recovery, and high cognitive-load proxies.

**Interpretive limits.** "Stress" here is a research aggregate that overlaps with but is not equivalent to clinical constructs like allostatic load, HPA reactivity, or DSM-defined anxiety states.

## 4. `environmental_burden`

**Construct.** Cumulative environmental stress on the participant from heat, humidity, and air quality, weighted by the participant's climate vulnerability.

**Direction.** Higher = worse.

**Observed proxies.**

- `heat_index`
- `nighttime_temperature_c`
- `aqi`
- 3-day cumulative heat exposure
- `heat_wave_flag`

**Expected forcing.** This is the most exogenous coordinate. It tracks the Environmental Physiological Load (EPL) scalar defined in `paper/mechanistic_formalism.md` and modulates the dynamics of all the other coordinates.

**Interpretive limits.** Climate exposure is approximated from coarse daily summaries, not microenvironment sensing. Indoor versus outdoor exposure is not disambiguated.

## 5. `behavioral_instability`

**Construct.** Irregularity of the participant's mobility, phone, and screen rhythms relative to their personal baseline.

**Direction.** Higher = less stable.

**Observed proxies.**

- `mobility_radius_km` variance
- `location_entropy` (high entropy in this construct = behavioral irregularity)
- `phone_unlock_count` deviation
- `screen_time_minutes` deviation
- `behavioral_regularity` (negative; note the inversion in feature engineering)

**Expected forcing.** Rises with stress load and with weekly rhythm disruption. Partially predicts future regime shifts.

**Interpretive limits.** Behavioral instability is a population-level signal. Some participants have legitimately irregular schedules (shift workers, caregivers) for whom this construct is not a deviation from norm. Subgroup analyses must control for this.

## 6. `missingness_pressure`

**Construct.** State-dependent dropout risk. Higher values mean the participant is more likely to stop logging EMAs, charging their wearable, or producing usable passive data, and the missingness itself carries signal about state.

**Direction.** Higher = more dropout risk.

**Observed proxies.**

- modality-dropout entropy
- consecutive-missing-days run length
- missingness rate over a 14-day rolling window
- `total_missing_modalities`

**Expected forcing.** Rises with `stress_load` and `behavioral_instability`. Conditioning on missingness pressure recovers part of the signal that naïve imputation would erase.

**Interpretive limits.** Missingness is treated as informative, not random. This is appropriate for this kind of multimodal sensing but requires care. Any downstream analysis that conditions on observed-only days will be biased relative to the underlying population.

## Non-clinical framing

These six coordinates are constructs for modeling, not diagnostic labels. They permit interpretable counterfactual simulation and regime analysis. They do not assess disease, do not replace clinical evaluation, and are not validated for any individual decision-making. Outputs derived from them are research signals only.
