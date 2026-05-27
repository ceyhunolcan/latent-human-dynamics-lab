# Data card — Synthetic cohort v0.1

## Source

All data shipped and exercised in this repository are fully synthetic. They are produced by `src/data/synthetic_generator.py` under a parameterized hierarchical model calibrated to plausible human ranges. There are no real participants. There is no real biometric data, no real location data, no real self-report data.

## Composition

A default invocation generates 500 participants observed for 180 days, which is 90,000 person-days across 51 base columns. After feature engineering this expands to roughly 110 columns. The columns are organized into:

**Identifiers.** `participant_id`, `date`.

**Participant baselines.** `age`, `sex`, `chronotype`, `baseline_sleep_need`, `baseline_hrv`, `baseline_resting_hr`, `baseline_activity_level`, `baseline_screen_time`, `baseline_stress_sensitivity`, `baseline_climate_vulnerability`, `baseline_resilience`, `baseline_missingness_tendency`.

**Wearable physiology.** `sleep_duration`, `sleep_efficiency`, `sleep_midpoint`, `hrv_rmssd`, `resting_hr`, `daily_steps`, `active_minutes`, `recovery_score`, `stress_score`.

**Passive behavioral sensing.** `screen_time_minutes`, `phone_unlock_count`, `mobility_radius_km`, `location_entropy`, `behavioral_regularity`.

**Environmental exposure.** `temperature_c`, `nighttime_temperature_c`, `humidity`, `aqi`, `heat_index`, `heat_wave_flag`, `poor_air_quality_flag`.

**Self-report proxies.** `mood_score`, `fatigue_score`, `perceived_stress`, `energy_score`, `cognitive_load_proxy`.

**Latent ground truth.** `latent_autonomic_recovery`, `latent_circadian_alignment`, `latent_stress_load`, `latent_environmental_burden`, `latent_behavioral_instability`, `latent_missingness_pressure`. These exist because the generator is hierarchical, and they make the synthetic cohort useful for evaluating recovery of latent structure.

**Regime label.** `regime_label` $\in$ {`stable`, `stressed`, `dysregulated`, `recovery`}.

**Missingness indicators.** `missing_wearable_flag`, `missing_phone_flag`, `missing_survey_flag`, `total_missing_modalities`.

## Generation process

Each participant samples a baseline vector from a hierarchical prior. Latent state evolves under autoregressive dynamics with environmental and behavioral forcings, seasonal phase $\cos\!\bigl(2\pi (\text{day} - 200) / 365\bigr)$ for a northern-hemisphere reference, and per-participant noise. Observable channels are generated as noisy generalized-linear projections of the latent state plus participant baselines. Missingness is state-dependent: participants in stressed or dysregulated regimes have higher dropout probability. The regime label is assigned post-hoc from a hand-specified rule over the latent state so downstream classifiers have a deterministic target.

## Bias and representational limits

The generator is a deliberate fiction. It does not reflect any real population's demographic, behavioral, or environmental distribution. Anyone porting to a real cohort should expect the marginal distributions, correlation structure, and especially the missingness pattern to differ. The synthetic-to-real similarity report in `src/evaluation/synthetic_to_real.py` reports KS, Wasserstein, correlation-matrix Frobenius, and missingness-pattern differences to make these gaps quantitative.

The northern-hemisphere seasonal phase will mis-align for southern-hemisphere cohorts. The climate-vulnerability prior is uniform across baseline strata, which over-represents climate-resilient phenotypes relative to many real cohorts. The missingness mechanism is monotone in stress, which is a simplification.

## Privacy and consent

No personal data are processed. No consent is required to use the synthetic generator. Anyone connecting real-cohort adapters is responsible for IRB / ethics review, informed consent, secure storage, and applicable privacy law (HIPAA, GDPR, local regulations).

## Recommended uses

Methodological development, dynamical-systems analysis, counterfactual-engine testing, robustness studies, benchmarking. Not for any clinical study, regulatory submission, or product feature delivered to end users.
