# Ethics

This is a research framework for modeling human physiological, behavioral, and environmental dynamics. Even in its synthetic form, it raises ethical considerations that anyone planning to extend it to real cohorts should think about before writing the data-loading code.

## Privacy and passive sensing

Passive sensing data are highly identifying. Even when individual channels are aggregated to daily summaries, the joint distribution of sleep midpoint, mobility entropy, and screen-time pattern is often unique to one person within a cohort of tens of thousands. If you are deploying a real-cohort version of this framework: store data only in encrypted, access-controlled environments; minimize the number of personnel with raw-data access; favor daily aggregates over raw waveforms whenever the science permits; refresh participant consent for new uses; document data-retention limits.

## Consent

Passive sensing tends to collect data that participants did not explicitly think about consenting to. Ambient temperature inferred from a phone. Location entropy that approximates social rhythm. Consent should be specific, plain-language, and updatable. Participants should be able to withdraw and have their data deleted without loss of benefit.

## Bias and fairness

Sensor accuracy varies with skin tone (photoplethysmography is well-documented to be noisier for darker skin), wrist size, fitness level, age, and disability status. Behavioral baselines vary with socioeconomic status, work patterns, and caregiving responsibilities. Climate vulnerability correlates with housing quality, neighborhood greenness, and ability to afford cooling. A model that performs equally on the cohort mean but worse on the most vulnerable subgroup is a worse model, not a neutral one. The evaluation module ships subgroup analyses and a climate-vulnerability tertile split precisely so these gaps are visible by default.

## Climate vulnerability

The framework treats environmental exposure as a directly modeled input. This is both its scientific contribution and its ethical risk surface. It makes it possible to differentiate "this person is climate-vulnerable" from "this person is dysregulated," and a downstream system could in principle use that difference to deny help, raise insurance premiums, or otherwise transfer climate risk back onto the people who already bear it. Climate-vulnerability scores are not surfaced as actionable user-facing outputs. The API returns latent components, not personalized risk labels.

## Non-diagnostic and non-clinical use

The framework is non-clinical. It does not output diagnoses, prognoses, or treatment recommendations. The output sanitizer (`src/safety/risk_language.py`, `src/safety/clinical_guardrails.py`) rewrites clinical vocabulary at every output boundary and enforces a canonical disclaimer on every API and dashboard response. Downstream consumers should preserve these guardrails. Anyone intending to develop a clinical product from this work would need to seek the appropriate regulatory pathway (FDA Software as a Medical Device, EU MDR, equivalents) under medical-device quality systems.

## Human oversight

Models of human state in the wild interact with people whose autonomy and self-understanding can be affected by what the model implies about them. Any deployed extension of this work should keep a clinically and ethically competent human in the loop for any decision that could affect a participant's care, employment, insurance, or housing. The framework's outputs are inputs to human judgement. They are not substitutes for it.

## Dual-use considerations

A model that infers stress load and behavioral instability from passive sensing is also a model that infers vulnerability. Refuse to deploy this framework in contexts where the inferred vulnerability could be turned against participants: workplace surveillance, parole monitoring, insurance underwriting, immigration enforcement, and similar settings.
