# Anticipated reviewer questions and responses

The following are the questions a careful peer reviewer would raise on this work. Honest answers, not defensive ones.

## R1: "Everything in your paper is synthetic. Why should I believe any of this generalizes?"

It doesn't, and the limitations section says so. The contribution is a framework: a particular structural decomposition of the modeling problem (latent state-space with environmental forcing, missingness as a modeled signal, counterfactual perturbations as an additive latent operator), a reproducible reference implementation, and a synthetic-to-real similarity report that makes the transfer question quantitative. The right reviewer test is whether the structural choices are useful, not whether the numbers transfer. A pre-registered real-cohort study (`paper/preprint_outline.md`) is the proposed next step.

## R2: "Your counterfactual operator is just a hand-coded direction vector with geometric decay. How is this 'causal' in any meaningful sense?"

It is causal under a stated assumption. The assumption is explicit in `paper/mechanistic_formalism.md`: the perturbation acts additively on the latent state with a fixed direction and a participant-specific resilience half-life. The assumption is wrong in detail and probably right in shape. Outputs are simulated trajectories, not effect estimates. A stronger version would estimate the direction vector and decay from a sub-cohort with a known intervention (e.g. a sleep-extension study), which is a planned next step.

## R3: "Why six latent dimensions? Why those names?"

The dimension count is empirical: across the synthetic generator's design space, six axes were sufficient to recover regime structure with cluster separation ≥ 0.4 silhouette, while five was not. The axis names are chosen for interpretability. They map to phenomena clinicians and behavioral scientists reason about anyway, but the axes themselves are constrained only by the encoder's regularization, not by supervision. A real-cohort fit may rotate them. The framework supports re-naming via the `LATENT_DIM_NAMES` constant.

## R4: "How do you handle the fact that more vulnerable participants also drop out more often, so your missingness mechanism is exactly the worst-case for fairness?"

This is the central reason the framework treats missingness as a modeled signal rather than as a nuisance to impute away. The `missingness_pressure` latent axis carries information about the participant's interaction with the sensing system, which is biased by exactly the demographics one should be worried about. The robustness module includes a missingness stress test up to 60% induced dropout and a per-subgroup performance table. The issue is not solved. But it is less ignored than the literature standard.

## R5: "Have you established identifiability of the dynamics drift?"

No, and that is in `paper/limitations.md`. A formal identifiability analysis is on the list of upgrades for the next iteration. Until it is done, posterior intervals on the drift coefficients under the Bayesian extension are wide, and there is no specific quantitative claim about $f_\theta$ that depends on its coefficients being uniquely recovered.

## R6: "Your environmental forcing weights look hand-picked. Why those numbers?"

They are literature-derived defaults, exposed in `configs/dynamics.yaml`, with the rationale documented in `paper/mechanistic_formalism.md`. The decision to weight nighttime heat at parity with daytime heat is the most consequential one and it reflects the literature on impaired thermoregulation during sleep and on excess mortality during high-nighttime-temperature heatwaves. In a real-cohort fit these would be absorbed into a learned per-subgroup forcing matrix rather than held fixed.

## R7: "The dashboard and API surface uncertainty estimates. Are they calibrated?"

Calibration curves and expected calibration error are reported in the evaluation module. On the synthetic cohort the dynamics prior's one-step uncertainty is well-calibrated. The downstream classification heads' uncertainty depends on the chosen architecture and is not calibrated by default. Per-dataset temperature scaling on a held-out partition is recommended before any downstream use.

## R8: "What stops someone from using this as a triage tool?"

A combination of things: an explicit non-clinical disclaimer on every API and dashboard response, automatic sanitization of clinical vocabulary at every output boundary, a model card that names clinical and triage uses as out-of-scope, and a license that names medical-device use as out-of-scope. None of these are technically enforced. A determined misuser can rewrite the disclaimer. Documentation and the research community's norms do the rest.

## R9: "What's the smallest cohort on which a real-cohort fit would be informative?"

For a registration-quality fit, $n \geq 200$ participants with $T \geq 120$ days, with explicit oversampling of the most vulnerable subgroups. Below that, the personalization layer's hierarchical pooling is doing most of the work and the per-participant estimates become unreliable.

## R10: "Why release this at all?"

Because the current state of passive-sensing modeling research is dominated by single-shot regressions onto questionnaire outcomes that do not respect the temporal structure of the underlying biology, and because the alternative (keeping multimodal state-space methods inside individual labs) slows the field and concentrates capability. The release is structured to make misuse harder than the default and useful methodological extension easier than the default.
