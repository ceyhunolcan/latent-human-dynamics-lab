# Preprint outline

**Title.** Latent Human Dynamics: A Multimodal State-Space Framework for Modeling Physiological and Behavioral Trajectories Under Environmental Perturbation

**Working venue targets.** Computational medicine, biomedical informatics, machine learning for health, climate-and-health interdisciplinary venues.

## Structure

**1. Introduction.** Passive sensing produces dense, longitudinal, multimodal data, but most modeling work in this space ignores temporal structure and is not equipped for counterfactual reasoning. Motivate the latent state-space approach. State the three claims of the paper: a structured decomposition (encoder → dynamics → perturbation operator → guardrails), a reference implementation, and a falsifiable path to external validation.

**2. Related work.** State-space models in physiology. Multimodal representation learning. Counterfactual prediction under temporal dynamics. Passive sensing and digital phenotyping. Environmental epidemiology and heat health. Uncertainty quantification for neural time-series models.

**3. Methods.** Drawn from `paper/methods.md` and `paper/mechanistic_formalism.md`. Subsections: cohort and data streams, feature engineering, latent state encoder, dynamics on the latent manifold, counterfactual perturbations, uncertainty, evaluation.

**4. Synthetic-cohort experiments.** Recovery of latent structure (does the encoder recover the generator's axes?). Regime detection accuracy. One-step and multi-step trajectory RMSE. Effect-size accuracy of counterfactual perturbations under known ground truth. Robustness under induced missingness up to 60%. Subgroup performance disaggregated by climate-vulnerability tertile.

**5. Real-cohort transfer demonstration.** Run the synthetic-to-real similarity report on at least one publicly available passive-sensing cohort (StudentLife, WESAD, or an Apple Health export). Report distributional differences, missingness-pattern differences, and re-fit the dynamics drift coefficients with hierarchical pooling. This is the section that turns the framework from a demonstration into a tool.

**6. Discussion.** What the framework cannot yet do. The role of synthetic-cohort prototyping in computational medicine. The case for treating missingness as a modeled signal. The ethical case for not surfacing climate-vulnerability scores as user-facing labels.

**7. Limitations.** Drawn from `paper/limitations.md`.

**8. Conclusion.** Multimodal state-space models with explicit counterfactual operators are a productive way to organize modeling work in passive sensing, provided the assumptions are made explicit, the missingness is treated as a modeled signal, and the guardrails against clinical overclaiming are baked in rather than bolted on.

## Pre-registration

Before any real-cohort number is generated, pre-register: the latent dimension count, the encoder regularization hyperparameters, the perturbation operator direction vectors and resilience half-life prior, the held-out evaluation cohort, the primary and secondary outcomes, the subgroup splits, the criteria for declaring negative results. Pre-registration goes on OSF or equivalent.

## Artifacts to release alongside the preprint

Code (this repository, archived to Zenodo with a DOI). Synthetic cohort (the deterministic output of the generator at fixed seed). Trained checkpoints for the dynamics model and downstream heads on the synthetic cohort. Notebooks reproducing every figure. A reviewer-response-simulation document (`paper/reviewer_response_simulation.md`). Model card, data card, ethics statement, limitations document.

## Author contributions and conflicts

To be filled in by the team. The non-clinical / non-medical-device clause in the license applies to all contributors.
