# Changelog

Significant changes, roughly in reverse chronological order. The project has gone through nine verification passes that turned up twenty-one bugs; the highlights are below. Format is loosely [Keep a Changelog](https://keepachangelog.com/) but adapted to a research-prototype rhythm.

## Unreleased

The big change in this stretch is wiring up a real-data path. The framework used to be synthetic-only; it now ingests StudentLife (Dartmouth, 2013) into the canonical daily-cohort schema and can compare any real cohort against the generator. The work for that exposed one fairly nasty bug (#22 below) that no synthetic-data probe could have caught.

### Real-data integration

- `scripts/ingest_studentlife.py` walks an unpacked StudentLife directory and aggregates the sensing streams (activity, conversation, dark, phonelock, GPS) and EMA responses (stress, mood, sleep) into one per-participant daily CSV matching the canonical schema. Unobserved channels (HRV, RHR, temperature, AQI — StudentLife doesn't collect these) come through as NaN, which the rest of the pipeline now tolerates.
- `scripts/compare_to_synthetic.py` takes any canonical-schema CSV and generates a matching synthetic cohort, then runs `synthetic_to_real_report` to produce a markdown summary, a JSON metrics file, and a 6-panel distribution comparison figure. Useful for "does the generator look anything like this real cohort?"

### Plumbing and ergonomics

- `scripts/smoke_test.py` runs the whole pipeline in ten labeled stages and finishes in under a second. Useful for CI and for verifying a fresh install before you do anything heavier.
- `src/utils/health_check.py` checks required deps, optional deps, paths, core module loads, and that the safety guardrail still sanitizes clinical language. Runs in milliseconds. Available as `python -m utils.health_check` or `make health`.
- `src/utils/pipeline_summary.py` records per-stage row/column counts, NaN rates, and timing. Wired into `run_pipeline.py`; every run now leaves a `results/tables/pipeline_summary.json` audit trail.
- `Makefile` with sensible defaults: `make smoke`, `make pipeline`, `make all`, `make ingest-studentlife DIR=...`, `make compare CSV=...`.
- GitHub Actions workflow at `.github/workflows/ci.yml` running pytest plus the smoke test on Python 3.10/3.11/3.12.

### YAML configs actually drive things now

This was a long-standing architectural gap — `load_default_config()` would load the YAML files but nothing downstream consumed them, so editing `configs/dynamics.yaml` had zero effect. Fixed:

- `utils.config.get_dynamics_settings()`, `get_epl_weights()`, `get_perturbation_defaults()` pull values from the config stack with code-default fallback.
- `LatentDynamicsModel.from_config()` reads `configs/dynamics.yaml` and produces a model configured from it.
- `--config` flag on `train_dynamics_model.py` lets you swap in a custom YAML at the command line.
- `--health` flag on `run_pipeline.py` runs the health check before any slow stages.
- `--perturbations` and `--magnitude` flags on `run_perturbation_demo.py` let you pick the subset and strength of interventions to compare.

### Bug fixes that mattered

**Bug 22** — `predict_next_state` silently propagated NaN env/behavior inputs to the latent state. Repro: `predict_next_state(z0=zeros, beh=zeros, env=[NaN]*4)` returned `[NaN, NaN, NaN, NaN, NaN, NaN]`. This made the framework unusable on any real cohort with missing days (i.e., all of them — StudentLife has no environmental data at all). Fix: impute NaN env/behavior with the normalization-reference values (`[18°C, 14°C, 50 AQI, 0]`), so the post-z-score signal contribution is zero, which is the right epistemic stance for "no observation." NaN in `Z_t` itself still raises loudly, because that means the encoder upstream failed and there's no sensible neutral.

**Bug 21** — `engineer_all_features` crashed with `ValueError: No objects to concatenate` on an empty cohort. The error came from deep inside pandas `groupby.transform()`, which makes it useless for debugging. Fixed with an empty-cohort short-circuit at the top of the function.

**Bug 20** — `NaN` for `vulnerability_coefficient` silently produced an all-NaN next state. Now raises `ValueError("vulnerability_coefficient must be finite; got nan. Use 1.0 for default.")`.

**Bug 19** — `configs/dynamics.yaml` said `noise_sigma: 0.05` but the paper and code both used σ=0.01. Aligned the YAML.

**Bug 18** — `configs/dynamics.yaml` used `heatwave_indicator` as a key but the EPL function looked for `heatwave`. Mismatched silently. Renamed and added a comment tying it to the code constant.

**Bug 17** — `PerturbationSpec(magnitude=NaN)` produced an all-NaN counterfactual trajectory without warning. `resolved_magnitude()` now rejects non-finite values.

**Bug 16** — Unknown perturbation type error message didn't list valid alternatives, so users had to grep the source to find a typo. Now includes the full list.

**Bug 15** — The Streamlit dashboard's Regime Timeline section would have crashed for every user. Three compounding issues: the dysregulated centroid wasn't being passed to the warning function (so `distance_to_dysregulated` was None), the warning signals are per-day arrays not scalars (so `:.3f` formatting raised `TypeError: unsupported format string passed to numpy.ndarray.__format__`), and `None`-format raised separately. Fixed by passing the centroid and adding a `_last(arr)` helper that returns the most recent finite value or NaN.

**Bug 14** — The `/detect-regime` API silently dropped one of its documented signals (`distance_to_dysregulated`) because the centroid was never passed through `critical_transition_warning_score`. Now extracts the centroid from the regime detector and passes it.

**Bug 13** — `recovery_half_life` was numerically wrong on clean inputs. The function shifted by `s.min() - 1e-3` before taking log, which destroys the exponential structure: $\log(A e^{-\lambda t} - A_{\min} + \varepsilon)$ is not linear in $t$. On a clean signal with true τ=3 days, the estimator returned 4.65. Fixed by logging directly when the input is positive, and only shifting when there are non-positive values to handle. Now recovers τ=3.0 exactly.

**Bug 12** — `transition_matrix` returned a bare ndarray but both callers (dashboard and notebook 04) expected `(matrix, regimes)`. Notebook 04 crashed on `ValueError: too many values to unpack`. Fixed to return both, which is also the more useful API.

**Bug 10** — `LogisticRegression(multi_class="auto")` was removed in scikit-learn 1.7. Dropped the kwarg.

**Bug 7 (the biggest)** — The dynamics model assumed z-scored inputs but every caller in the repo was passing raw values (temperature in °C, screen time in minutes, etc.). One-step latent RMSE was 6.6. Added `normalize_inputs=True` as the default; RMSE dropped to 0.87.

The other early-pass bugs (#1-9) are mostly in the same shape: schema mismatches between the generator and the feature engineering layer, edge cases in the encoder for short windows, sign errors in perturbation operators, etc. The full list is in the git history.

### Verification at the end of this stretch

- Test suite: 64 passing, 0 failing, 1 skipped (the skipped test needs `fastapi`).
- Smoke test: 10 stages, under a second wall-clock.
- All 6 notebooks execute end-to-end without errors.
- Full pipeline (1800 rows, 30 participants, 60 days) produces RMSE ≈ 0.87 against the dynamics prior.
- AI-tells regression: zero matches across the repo for the project's watchlist of giveaway phrases.

## 0.1.0 — Initial structure

The first cut, before the verification passes started.

- Synthetic generator producing the canonical 51-column daily-cohort schema.
- Classical and neural latent state encoders mapping multimodal observations into a 6-dimensional latent space (autonomic recovery, circadian alignment, stress load, environmental burden, behavioral instability, missingness pressure).
- Latent dynamics model with environmental and behavioral forcings.
- Regime detector (k-means in latent space with a stability score and a transition matrix).
- Counterfactual perturbation engine for seven canonical interventions (sleep extension, screen reduction, exercise boost, cooling, air-quality improvement, heat-wave shock, combined resilience protocol).
- Early warning score combining variance, autocorrelation, instability index, and distance to the dysregulated centroid.
- Energy landscape estimation via 2D kernel density.
- Resilience model with per-participant half-life estimation and profile classification.
- Safety guardrail rewriting clinical language and attaching a non-clinical disclaimer to every output.
- FastAPI service exposing `/encode`, `/detect-regime`, and `/simulate-perturbation`.
- Streamlit dashboard with six sections.
- Nine paper documents covering motivation, mechanistic formalism, methodology, limitations, ethics, and validation.
- Six worked-example notebooks and five reproducible figures.
