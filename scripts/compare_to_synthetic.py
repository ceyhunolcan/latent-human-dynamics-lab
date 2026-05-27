"""Compare a real cohort CSV against the synthetic generator.

Run from repo root:

    python scripts/compare_to_synthetic.py /path/to/real_daily.csv

The input CSV should follow the canonical schema (see
src/data/synthetic_generator.py). Columns missing in the real data are
ignored — only the overlap is compared.

Output:
  - results/tables/synthetic_vs_real_report.md  (human-readable)
  - results/tables/synthetic_vs_real_metrics.json (programmatic)
  - results/figures/synthetic_vs_real_*.png (per-channel distributions)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.synthetic_generator import generate_synthetic_cohort
from evaluation.synthetic_to_real import synthetic_to_real_report
from utils.logging import get_logger
from utils.paths import RESULTS_DIR, ensure_dir

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare real cohort CSV against the synthetic generator.")
    parser.add_argument("real_csv", type=str, help="Path to real cohort CSV in canonical schema")
    parser.add_argument(
        "--n-participants", type=int, default=None,
        help="Generate synthetic cohort with this many participants (default: match real)",
    )
    parser.add_argument(
        "--n-days", type=int, default=None,
        help="Generate synthetic cohort with this many days (default: match real)",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    real_path = Path(args.real_csv).expanduser().resolve()
    if not real_path.exists():
        logger.error("Real CSV not found: %s", real_path)
        return 1

    logger.info("Loading real cohort from %s", real_path)
    real = pd.read_csv(real_path, parse_dates=["date"])
    logger.info("Real cohort: %d rows × %d cols, %d participants, %d days span",
                len(real), real.shape[1],
                real["participant_id"].nunique(),
                (real["date"].max() - real["date"].min()).days)

    # Match the synthetic cohort to the real one's shape
    n_p = args.n_participants or real["participant_id"].nunique()
    days_span = (real["date"].max() - real["date"].min()).days + 1
    n_d = args.n_days or min(days_span, 180)

    logger.info("Generating matching synthetic cohort: %d participants × %d days", n_p, n_d)
    synth = generate_synthetic_cohort(n_participants=n_p, n_days=n_d, seed=args.seed)

    # Identify overlapping columns
    real_cols = set(real.columns)
    synth_cols = set(synth.columns)
    overlap = sorted(real_cols & synth_cols - {"participant_id", "date"})
    logger.info("Overlapping columns to compare: %d", len(overlap))
    for c in overlap:
        real_pct = 100 * real[c].notna().mean()
        synth_pct = 100 * synth[c].notna().mean()
        logger.info("  %-32s  real %5.1f%% non-NaN, synth %5.1f%% non-NaN",
                    c, real_pct, synth_pct)

    # Run the canonical similarity report
    logger.info("Running synthetic_to_real_report ...")
    report = synthetic_to_real_report(synth[overlap], real[overlap])

    # Persist
    ensure_dir(RESULTS_DIR / "tables")
    md_path = RESULTS_DIR / "tables" / "synthetic_vs_real_report.md"
    md_path.write_text(report.as_markdown())
    logger.info("Markdown report → %s", md_path)

    json_path = RESULTS_DIR / "tables" / "synthetic_vs_real_metrics.json"
    metrics_dict = {
        "correlation_frobenius": float(report.correlation_frobenius),
        "distribution_table": report.distribution_table.to_dict(orient="records"),
        "missingness_table": report.missingness_table.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(metrics_dict, indent=2, default=float))
    logger.info("Metrics JSON → %s", json_path)

    # Per-column distribution plots for the highest-overlap channels
    fig_dir = RESULTS_DIR / "figures"
    ensure_dir(fig_dir)
    top_channels = sorted(
        overlap,
        key=lambda c: -(real[c].notna().mean() + synth[c].notna().mean()),
    )[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, top_channels):
        real_vals = real[col].dropna()
        synth_vals = synth[col].dropna()
        if len(real_vals) == 0 or len(synth_vals) == 0:
            ax.set_title(f"{col}\n(no overlap)")
            continue
        bins = np.linspace(
            min(real_vals.min(), synth_vals.min()),
            max(real_vals.max(), synth_vals.max()),
            30,
        )
        ax.hist(synth_vals, bins=bins, alpha=0.5, label=f"synthetic (n={len(synth_vals)})", color="#e41a1c")
        ax.hist(real_vals, bins=bins, alpha=0.5, label=f"real (n={len(real_vals)})", color="#377eb8")
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Synthetic vs real cohort — top 6 channels by coverage", fontsize=12)
    fig.tight_layout()
    fig_path = fig_dir / "synthetic_vs_real_distributions.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Distribution figure → %s", fig_path)

    print()
    print(f"Comparison complete.")
    print(f"  Channels compared : {len(overlap)}")
    print(f"  Report (md)       : {md_path}")
    print(f"  Metrics (JSON)    : {json_path}")
    print(f"  Distributions     : {fig_path}")
    print()
    print("(research prototype, non-clinical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
