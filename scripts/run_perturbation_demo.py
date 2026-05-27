"""Counterfactual perturbation demo.

Picks one participant from the processed cohort, encodes their recent latent
trajectory, and simulates three contrasting perturbations (cooling, sleep
extension, heat-wave shock). Writes per-perturbation pathway summaries plus
a comparison plot to ``results/figures/``.

Run from repo root::

    python scripts/run_perturbation_demo.py
    python scripts/run_perturbation_demo.py --participant P0042
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from utils.logging import get_logger  # noqa: E402
from utils.paths import PROCESSED_DIR, RESULTS_DIR, ensure_dir  # noqa: E402
from features import engineer_all_features  # noqa: E402
from states.latent_state_encoder import (  # noqa: E402
    LATENT_DIM_NAMES,
    encode_latent_states_classical,
)
from dynamics.transition_model import LatentDynamicsModel  # noqa: E402
from counterfactuals.perturbation_engine import (  # noqa: E402
    PerturbationSpec,
    simulate_perturbation,
)
from counterfactuals.intervention_simulator import summarise_intervention_pathway  # noqa: E402

logger = get_logger(__name__)


def _modality_matrices(df: pd.DataFrame):
    def pick(cols):
        present = [c for c in cols if c in df.columns]
        return df[present].to_numpy(dtype=float, na_value=0.0) if present else np.zeros((len(df), len(cols)))

    W = pick(["sleep_duration_hours", "hrv_rmssd", "resting_hr", "daily_steps", "recovery_score"])
    B = pick(["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"])
    C = pick(["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"])
    M = pick(["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"])
    P = pick(["baseline_hrv", "baseline_resting_hr", "baseline_climate_vulnerability", "baseline_resilience"])
    return W, B, C, M, P


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a 3-panel counterfactual perturbation demo.",
    )
    parser.add_argument("--participant", type=str, default=None,
                        help="Participant ID (default: first alphabetical)")
    parser.add_argument("--horizon", type=int, default=14,
                        help="Forecast horizon in days (default: 14)")
    parser.add_argument(
        "--perturbations", type=str, default="cooling,sleep_extension,heat_wave_shock",
        help="Comma-separated perturbation types to compare "
             "(default: cooling,sleep_extension,heat_wave_shock)",
    )
    parser.add_argument(
        "--magnitude", type=float, default=None,
        help="Override magnitude for all selected perturbations (default: per-type default)",
    )
    args = parser.parse_args()

    processed_path = PROCESSED_DIR / "processed_features.csv"
    if not processed_path.exists():
        logger.error("Processed features missing. Run scripts/run_pipeline.py first.")
        return 1

    df = pd.read_csv(processed_path, parse_dates=["date"])
    if "sleep_duration_hours" not in df.columns:
        df = engineer_all_features(df)

    pid = args.participant or sorted(df["participant_id"].unique())[0]
    logger.info("Demo participant: %s", pid)
    pdf = df[df["participant_id"] == pid].sort_values("date").reset_index(drop=True)
    if len(pdf) < 14:
        logger.error("Participant %s has only %d days, need at least 14", pid, len(pdf))
        return 1

    # Encode the participant's window. We re-fit the classical encoder on the
    # full cohort so the projection axes are stable across runs.
    W, B, C, M, P = _modality_matrices(df)
    Z = encode_latent_states_classical(W, B, C, M, P).latent
    pZ = Z[df["participant_id"].values == pid]
    z0 = pZ[-1]

    env_cols = ["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"]
    beh_cols = ["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"]
    env_last = pdf[env_cols].tail(1).to_numpy(dtype=float, na_value=0.0)
    beh_last = pdf[beh_cols].tail(1).to_numpy(dtype=float, na_value=0.0)
    env_forecast = np.tile(env_last, (args.horizon, 1))
    beh_forecast = np.tile(beh_last, (args.horizon, 1))

    dynamics = LatentDynamicsModel(rng_seed=17)
    # Parse the comma-separated perturbation list with whitespace tolerance
    from counterfactuals.perturbation_engine import available_perturbations
    requested = [s.strip() for s in args.perturbations.split(",") if s.strip()]
    valid = available_perturbations()
    bad = [p for p in requested if p not in valid]
    if bad:
        logger.error(
            "Unknown perturbation type(s): %s. Valid options: %s",
            bad, valid,
        )
        return 1
    perturbations = requested
    results = {}

    n_panels = len(perturbations)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 4.5), sharey=True, squeeze=False,
    )
    axes = axes.flatten()
    for ax, ptype in zip(axes, perturbations):
        spec = PerturbationSpec(
            perturbation_type=ptype, magnitude=args.magnitude,
            horizon_days=args.horizon,
        )
        result = simulate_perturbation(z0, env_forecast, beh_forecast, spec, dynamics_model=dynamics)
        results[ptype] = result.to_dict()

        base = np.array(result.baseline_trajectory)
        cf = np.array(result.counterfactual_trajectory)
        t = np.arange(len(base))
        for i, name in enumerate(LATENT_DIM_NAMES):
            ax.plot(t, base[:, i], "--", color=f"C{i}", alpha=0.4)
            ax.plot(t, cf[:, i], "-", color=f"C{i}", label=name if ax is axes[-1] else None)
        ax.set_title(f"{ptype}  (mag={result.magnitude:.2f})")
        ax.set_xlabel("Days from intervention")
        ax.axhline(0, color="grey", linewidth=0.5)

    axes[0].set_ylabel("Latent value")
    axes[-1].legend(loc="upper right", fontsize=7, ncol=1)
    fig.suptitle(
        f"Counterfactual perturbations — {pid}  "
        "(dashed = baseline, solid = counterfactual)",
        fontsize=12,
    )
    fig.tight_layout()

    fig_dir = RESULTS_DIR / "figures"
    ensure_dir(fig_dir)
    fig_path = fig_dir / f"perturbation_demo_{pid}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    table_dir = RESULTS_DIR / "tables"
    ensure_dir(table_dir)
    json_path = table_dir / f"perturbation_demo_{pid}.json"
    json_path.write_text(json.dumps(results, indent=2, default=float))

    print(f"\nPerturbation demo complete for {pid}.")
    print(f"  Figure : {fig_path}")
    print(f"  JSON   : {json_path}\n")
    for ptype, res in results.items():
        print(f"--- {ptype} (magnitude {res['magnitude']:.2f}, horizon {res['horizon_days']}d) ---")
        print(res["pathway_explanation"])
        proxies = res["proxy_delta_mean"]
        top = sorted(proxies.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        for name, val in top:
            print(f"    Δ {name:<30s} = {val:+.4f}")
        print(f"    {res['disclaimer']}")
        print()
    print("(research prototype, non-clinical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
