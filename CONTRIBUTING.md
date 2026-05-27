# Contributing

A few notes if you want to send a PR or open an issue.

## What this is and isn't

The repo is a research prototype for modeling physiological, behavioral, and environmental dynamics in daily-cadence passive-sensing data. It's not a clinical tool, and the safety layer in `src/safety/` enforces that — every API and dashboard response goes through a sanitizer that rewrites clinical language and attaches a non-clinical disclaimer. Don't weaken that layer. If a change you want to make would let clinical phrasing leak through, that's a no.

## Getting set up

```bash
git clone https://github.com/ceyhunolcan/latent-human-dynamics-lab.git
cd latent-human-dynamics-lab
pip install -e .
pip install pytest
make health
make smoke
```

If `make health` is green and `make smoke` prints "All 10 stages passed," you're good.

## How to send a change

For anything more than a typo fix, open an issue first describing what you want to do and why. That's mostly so I can flag if the change conflicts with something already in flight, or if there's a simpler path. For science contributions — new generator couplings, dynamics terms, evaluation metrics — include a paragraph about why you think the change is right (papers, empirical observation from another cohort, etc.).

When you send the PR:

- Pass `make check` locally first. That runs health + smoke + the test suite.
- If you changed a formula, update both the code and the matching paper section under `paper/`. Drift between the two is how the project accumulates technical debt.
- If you fixed a bug, add a regression test. The convention is to name it something like `test_<what_used_to_break>_now_works` and add a one-line docstring explaining the original failure mode. There are 20+ of these in `tests/` already; match that pattern.

## Things to know about the code

The repo's been through about nine verification passes and twenty-one documented bug fixes, so a few patterns are load-bearing:

The canonical column schema in `src/data/synthetic_generator.py` is treated as a contract. Adding columns is fine, renaming or removing them breaks every adapter and evaluation downstream. If you genuinely need to rename one, do it in a dedicated PR.

NaN propagation is something I've fought with repeatedly. The pattern that won: impute NaN to the normalization reference at the dynamics layer (so missing data → no contribution), and raise loudly if NaN shows up in the latent state itself (because that means the encoder upstream already failed). Match that pattern if you add new transformations.

The YAML configs in `configs/` actually drive behavior now (`LatentDynamicsModel.from_config()` reads them). If you add a constant that a user might want to tune, expose it through the config rather than hardcoding it.

Heavy dependencies (torch, fastapi, streamlit) are optional and guarded by `try/except ImportError`. Keep them that way. The smoke test should pass on a CPU-only laptop with just numpy/pandas/sklearn/scipy/matplotlib/pyyaml.

## Working with a new real-cohort dataset

If you want to add support for a dataset besides StudentLife, the path is:

Write an ingestion script under `scripts/` that aggregates the raw streams into one CSV matching the canonical schema. Use `scripts/ingest_studentlife.py` as a reference. Unobserved channels go in as NaN — the dynamics layer handles that now. Then write an adapter under `src/adapters/` for the column-level mapping (mostly bookkeeping, but it's where dataset-specific quirks live). Finally, run `python scripts/compare_to_synthetic.py your_cohort.csv` and look at where the generator diverges from real data. That divergence is the most interesting thing you'll find.

## Reporting bugs

Open an issue with what you ran, what you expected, what happened instead, and the output of `python -m utils.health_check` if it's relevant. "This used to work" reports are taken seriously — there's a regression test for every documented bug, so a real regression would mean a test got bypassed.

## Things I'd rather you didn't contribute

Anything that weakens the clinical-language guardrail. Anything that requires a GPU just to run the smoke test. Marketing language — the framing is research, not medicine, and words like "cure" or "diagnose" or "patient" shouldn't appear in code or docs even though the sanitizer would catch them at output time. Required dependencies on heavy ML frameworks; keep torch/jax/etc. optional.

## License

Contributions get licensed under the same MIT terms as the rest of the repo. See `LICENSE`.
