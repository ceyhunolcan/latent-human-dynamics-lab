"""End-to-end synthetic data pipeline.

Generates the synthetic cohort, validates it, engineers features, and
writes the processed dataframe to ``data/processed/processed_features.csv``.

Run from repo root::

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --participants 500 --days 180
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.synthetic_generator import generate_synthetic_cohort, save_synthetic_data  # noqa: E402
from data.validation import validate_cohort  # noqa: E402
from data.preprocessing import full_preprocess  # noqa: E402,F401  (re-exported for convenience)
from features import engineer_all_features  # noqa: E402
from utils.logging import get_logger  # noqa: E402
from utils.paths import PROCESSED_DIR, ensure_dir  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic data pipeline.")
    parser.add_argument("--participants", type=int, default=None,
                        help="Override cohort.n_participants from config (default: 500 or config value)")
    parser.add_argument("--days", type=int, default=None,
                        help="Override cohort.n_days from config (default: 180 or config value)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config (default: 17 or config value)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--no-save", action="store_true", help="Skip writing the synthetic CSV")
    parser.add_argument(
        "--health", action="store_true",
        help="Run a health check first (imports, deps, paths) and exit early if anything fails.",
    )
    args = parser.parse_args()

    # Health check gate: useful in CI or right after install to catch broken
    # environments before the full pipeline runs.
    if args.health:
        from utils.health_check import health_check
        report = health_check(verbose=True)
        if not report.ok:
            logger.error("Health check failed (%d failures). Aborting pipeline.", len(report.failed))
            return 3
        logger.info("Health check passed: %s", report.summary())

    # Load config (falls back to defaults if file missing)
    from utils.config import load_yaml, load_default_config
    from utils.paths import CONFIGS_DIR
    if args.config:
        cfg_default = load_yaml(args.config)
    else:
        full_cfg = load_default_config()
        cfg_default = full_cfg.get("default", {})

    # Resolve params: CLI > YAML > hardcoded fallback
    cohort_cfg = cfg_default.get("cohort", {})
    n_participants = args.participants or cohort_cfg.get("n_participants", 500)
    n_days = args.days or cohort_cfg.get("n_days", 180)
    seed = args.seed if args.seed is not None else cfg_default.get("seed", 17)

    logger.info(
        "Generating synthetic cohort: %d participants × %d days (seed=%d)",
        n_participants, n_days, seed,
    )
    from utils.pipeline_summary import PipelineSummary, StageTimer
    summary = PipelineSummary()

    empty_df = __import__("pandas").DataFrame()
    timer = StageTimer()
    cohort = generate_synthetic_cohort(
        n_participants=n_participants, n_days=n_days, seed=seed,
    )
    summary.record("generate", empty_df, cohort, timer.elapsed(),
                   notes=f"seed={seed}")
    logger.info("Generated %d rows × %d columns", *cohort.shape)

    logger.info("Validating cohort schema and value ranges")
    try:
        validate_cohort(cohort)
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        logger.error("Hint: if you're feeding real-cohort data through this pipeline, "
                     "use --no-save and adapt src/adapters/ to coerce columns to the expected schema.")
        return 2

    if not args.no_save:
        path = save_synthetic_data(cohort)
        logger.info("Saved synthetic cohort to %s", path)

    logger.info("Preprocessing (clean → sort → impute)")
    from data.preprocessing import clean_data, sort_by_participant_date, impute_safe_defaults

    timer = StageTimer()
    cohort_pre = cohort
    cohort = impute_safe_defaults(sort_by_participant_date(clean_data(cohort)))
    summary.record("preprocess", cohort_pre, cohort, timer.elapsed())

    logger.info("Engineering features (wearable, behavioral, climate, missingness, baseline)")
    timer = StageTimer()
    processed = engineer_all_features(cohort)
    summary.record("engineer", cohort, processed, timer.elapsed())

    ensure_dir(PROCESSED_DIR)
    out = PROCESSED_DIR / "processed_features.csv"
    processed.to_csv(out, index=False)
    logger.info("Wrote processed features → %s  shape=%s", out, processed.shape)

    # Save the pipeline summary alongside the processed features
    import json
    from utils.paths import RESULTS_DIR
    ensure_dir(RESULTS_DIR / "tables")
    summary_path = RESULTS_DIR / "tables" / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary.as_dict(), indent=2))
    logger.info("Pipeline stage summary → %s", summary_path)

    print("\nPipeline complete.")
    print(f"  Synthetic rows : {len(cohort)}")
    print(f"  Feature columns: {processed.shape[1]}")
    print(f"  Output         : {out}")
    print(f"  Stage summary  : {summary_path}")
    print()
    print(summary.as_markdown())
    print("\n(research prototype, non-clinical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
