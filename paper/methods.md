# Methods

## Cohort and data streams

The framework ingests four parallel daily streams per participant. Wearable physiology covers sleep duration and efficiency, sleep midpoint, HRV RMSSD, resting heart rate, daily steps, recovery score and stress score. Passive behavioral sensing covers screen time, phone unlock counts, mobility radius, location entropy and behavioral regularity. Environmental exposure covers daytime and nighttime temperature, humidity, AQI, heat index, heat-wave and poor-air-quality flags. Self-report or ground-truth labels cover mood, fatigue, perceived stress, energy, and a cognitive-load proxy. Each participant also carries a small panel of stable baselines: age, sex, chronotype, baseline HRV and resting HR, baseline screen time and activity level, stress sensitivity, climate vulnerability, resilience, missingness tendency.

The release ships a synthetic cohort generator that produces these streams under a parameterized hierarchical model. Latent biological state evolves under autoregressive dynamics with environmental and behavioral forcings, observable channels are noisy linear or generalized-linear projections of latent state, and missingness is state-dependent. The synthetic cohort is the input to all the examples and tests, and the same code paths accept real cohorts through the adapter layer in `src/adapters/`.

## Feature engineering

Features are engineered in a fixed order: missingness, then personal baselines, then wearable, then behavioral, then climate. The ordering matters. Missingness has to be computed before any imputation, because imputation by construction erases the dropout pattern that the framework treats as informative. Personal baselines (28-day rolling means, 21-day rolling z-scores, personal anomaly scores) have to be computed before group-level transforms so downstream rolling statistics operate on within-person deviations rather than between-person rank differences. The climate features include cumulative heat burden, nighttime heat stress, AQI burden, consecutive heat-wave exposure days, and the composite Environmental Physiological Load (EPL).

## Latent state encoder

The encoder maps a daily multimodal observation $x_t = (W_t, B_t, C_t, M_t, P)$ to a six-dimensional latent state $z_t \in \mathbb{R}^6$ with axes named autonomic recovery, circadian alignment, stress load, environmental burden, behavioral instability and missingness pressure. There are two encoders. The neural encoder (`MultimodalLatentStateEncoder`, PyTorch) is the one you want when you have time to train, and the classical encoder applies a per-modality PCA and then a final PCA over the concatenation. The classical encoder is used as a deterministic fallback when PyTorch is unavailable, and also by the API service so the system remains operable on CPU-only hosts without a trained checkpoint.

Training the neural encoder minimizes a four-term objective: reconstruction loss across modalities, next-state prediction loss, smoothness regularization that penalizes high curvature in $z_t$, and a contrastive trajectory loss that pulls together latent states from the same participant on neighboring days. A disentanglement penalty is added when supervised regime labels are available.

## Dynamics on the latent manifold

Transition model: $z_{t+1} = (1 - \kappa)\, z_t + \alpha \cdot F_{\text{env}}(C_t) + \alpha \cdot F_{\text{beh}}(B_t) + p_t + \epsilon_t$ with contraction $\kappa$, forcing scale $\alpha$, Gaussian residual $\epsilon_t$, and optional additive perturbation $p_t$. Environmental and behavioral forcings are deterministic projections from raw channels onto latent dimensions, so every coefficient in the drift is auditable. The implementation provides both a Euler integrator and an RK4 integrator, and `NeuralODEStep` exposes a learned residual when you want to override the parametric drift.

## Counterfactual perturbations

Seven perturbation primitives: `sleep_extension`, `screen_reduction`, `exercise_boost`, `cooling`, `air_quality_improvement`, `heat_wave_shock`, `combined_resilience_protocol`. Each takes a magnitude in natural units, a horizon, and a per-participant vulnerability coefficient. The perturbation operator returns an additive (horizon, 6) latent delta with geometric decay over a resilience half-life. The heat-wave operator ramps up rather than decaying. Counterfactual trajectories are produced by running the dynamics model twice (once unperturbed, once with the perturbation injected), and the latent delta is mapped back to observed proxies (sleep duration, HRV, RHR, recovery score, stress score, fatigue score, mood score, missingness) via a linear read-out matrix.

## Uncertainty and evaluation

MC-dropout and lightweight deep ensembles are exposed for predictive uncertainty. The evaluation module reports regression and classification metrics, calibration curves with expected calibration error, a missingness stress test across induced dropout rates, heat-wave and climate-vulnerability subgroup analyses, out-of-distribution environmental shock tests, representation-PCA and cluster-separation scores, and a synthetic-to-real similarity report (KS, Wasserstein, correlation Frobenius, missingness pattern) for porting to real cohorts.

## Reproducibility

Everything is deterministic. A single `seed=17` controls both the synthetic generator and the dynamics noise. The repository ships Docker, pinned `requirements.txt`, six executable notebooks, a smoke-test harness, and a small leaderboard CSV pre-populated for the baseline models on the synthetic cohort.
