# Figure gallery

Five canonical figures, all produced by `scripts/generate_figures.py` from the processed synthetic cohort.

## 1. `cohort_latent_state_distribution.png`

Histograms of each of the six latent coordinates across the cohort, six panels. Useful as a sanity check that the encoder is producing axes that are centered near zero and reasonably symmetric. Skew or bimodality on an axis usually means the cohort contains a subgroup whose state-space occupation differs systematically.

## 2. `regime_phase_diagram.png`

The latent cloud projected into a 2D PCA plane, colored by inferred regime (`stable`, `stressed`, `dysregulated`, `recovery`). The four clusters should be visually separable on the synthetic cohort. On a real cohort, expect more overlap.

## 3. `energy_landscape.png`

The pseudo-energy surface $E(z) = -\log p(z)$ over the same 2D PCA plane. Basins are common state configurations, ridges are rare or unstable ones. Single-basin landscapes (like the synthetic cohort) indicate a homogeneous cohort or insufficient sampling of dysregulated states.

## 4. `environmental_forcing_response.png`

Mean latent state on top-quintile EPL days versus bottom-quintile EPL days. The expected pattern: `environmental_burden` and `stress_load` rise in the high-EPL group while `autonomic_recovery` and `circadian_alignment` drop. Deviations from this pattern flag a generator or encoder issue.

## 5. `perturbation_pathways.png`

Per-perturbation latent delta over a 14-day horizon for a typical participant, eight panels (one per perturbation, plus an empty panel). Each curve is the counterfactual minus baseline trajectory of one latent dimension. Heat-wave shock should ramp up over time, the rest should decay.

## Demo figures

`scripts/run_perturbation_demo.py` produces one additional figure per participant: `perturbation_demo_<participant_id>.png`, with three side-by-side panels showing baseline (dashed) vs counterfactual (solid) for three contrasting perturbations: cooling, sleep extension, heat-wave shock.
