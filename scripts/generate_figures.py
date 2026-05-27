"""Generate the canonical figure set for the figure gallery.

Produces five figures into ``results/figures/``:

1. ``cohort_latent_state_distribution.png`` — joint and marginal distributions
   of the six latent dimensions across the cohort.
2. ``regime_phase_diagram.png`` — 2D PCA of the latent state cloud, coloured
   by inferred regime.
3. ``energy_landscape.png`` — pseudo-energy surface :math:`E(z) = -\\log p(z)`.
4. ``environmental_forcing_response.png`` — group-mean latent response to high
   environmental physiological load (EPL) days.
5. ``perturbation_pathways.png`` — average counterfactual trajectory for each
   of the seven perturbation types applied to a typical participant.

Run from repo root::

    python scripts/generate_figures.py
"""

from __future__ import annotations

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
from states.regime_detector import fit_regime_detector  # noqa: E402
from states.state_geometry import project_latent_2d  # noqa: E402
from states.energy_landscape import (  # noqa: E402
    estimate_energy_landscape,
    plot_energy_landscape,
)
from dynamics.transition_model import LatentDynamicsModel  # noqa: E402
from counterfactuals.perturbation_engine import (  # noqa: E402
    PerturbationSpec,
    available_perturbations,
    simulate_perturbation,
)

logger = get_logger(__name__)


def _modality_matrices(df):
    def pick(cols):
        present = [c for c in cols if c in df.columns]
        return df[present].to_numpy(dtype=float, na_value=0.0) if present else np.zeros((len(df), len(cols)))

    W = pick(["sleep_duration_hours", "hrv_rmssd", "resting_hr", "daily_steps", "recovery_score"])
    B = pick(["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"])
    C = pick(["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"])
    M = pick(["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"])
    P = pick(["baseline_hrv", "baseline_resting_hr", "baseline_climate_vulnerability", "baseline_resilience"])
    return W, B, C, M, P


def figure_latent_distribution(Z, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, (ax, name) in enumerate(zip(axes.flat, LATENT_DIM_NAMES)):
        ax.hist(Z[:, i], bins=60, color="steelblue", alpha=0.85, edgecolor="white")
        ax.set_title(name, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    fig.suptitle("Latent state distribution across the cohort", fontsize=13)
    fig.tight_layout()
    out = fig_dir / "cohort_latent_state_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_regime_phase_diagram(Z, regimes, fig_dir):
    Z2 = project_latent_2d(Z)
    palette = {
        "stable": "#4daf4a",
        "stressed": "#ff7f00",
        "dysregulated": "#e41a1c",
        "recovery": "#377eb8",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in sorted(set(regimes)):
        mask = np.array(regimes) == r
        ax.scatter(
            Z2[mask, 0], Z2[mask, 1], s=6, alpha=0.5, label=r,
            color=palette.get(r, "grey"),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent state phase diagram, coloured by regime")
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    out = fig_dir / "regime_phase_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_energy_landscape(Z, fig_dir):
    landscape = estimate_energy_landscape(Z, grid_size=80)
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_energy_landscape(landscape, ax=ax)
    ax.set_title(r"Pseudo-energy landscape  $E(z) = -\log p(z)$")
    fig.tight_layout()
    out = fig_dir / "energy_landscape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_environmental_response(df, Z, fig_dir):
    if "environmental_physiological_load" not in df.columns:
        return None
    epl = df["environmental_physiological_load"].to_numpy(dtype=float)

    # Use rank-based tertiles so ties at quantile boundaries don't produce
    # empty groups. EPL is a composite of a few discrete environmental
    # signals (heat-wave flag, AQI bucket) so many rows can sit at the
    # exact 20th-percentile value; strict comparisons would drop them all.
    epl_clean = epl[~np.isnan(epl)]
    rank = pd.Series(epl).rank(method="average", pct=True).to_numpy()
    high_mask = (rank >= 2 / 3) & (~np.isnan(epl))
    low_mask = (rank <= 1 / 3) & (~np.isnan(epl))

    if high_mask.sum() == 0 or low_mask.sum() == 0:
        # Pathological case (e.g. all-NaN). Skip the figure rather than
        # produce a misleading one.
        return None

    high_mean = Z[high_mask].mean(axis=0)
    low_mean = Z[low_mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LATENT_DIM_NAMES))
    width = 0.35
    ax.bar(x - width / 2, low_mean, width,
           label=f"low EPL (bottom tertile, n={low_mask.sum()})", color="#377eb8")
    ax.bar(x + width / 2, high_mean, width,
           label=f"high EPL (top tertile, n={high_mask.sum()})", color="#e41a1c")
    ax.set_xticks(x)
    ax.set_xticklabels(LATENT_DIM_NAMES, rotation=25, ha="right")
    ax.set_ylabel("Mean latent value (classical PCA encoder)")
    ax.set_title("Latent response to environmental physiological load")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    # Note: axis labels reflect the canonical taxonomy but the classical
    # PCA encoder produces unsupervised components — signs and exact
    # alignment can rotate. See paper/limitations.md (identifiability).
    fig.text(0.5, -0.05,
             "classical PCA encoder; axis alignment is unsupervised",
             ha="center", fontsize=8, style="italic", color="grey")
    fig.tight_layout()
    out = fig_dir / "environmental_forcing_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_perturbation_pathways(df, Z, fig_dir):
    # Pick a "typical" participant (closest to cohort centroid)
    centroid = Z.mean(axis=0)
    pids = df["participant_id"].values
    unique_pids = np.unique(pids)
    closest_pid = None
    best_dist = np.inf
    for pid in unique_pids:
        mask = pids == pid
        if mask.sum() < 14:
            continue
        d = np.linalg.norm(Z[mask].mean(axis=0) - centroid)
        if d < best_dist:
            best_dist = d
            closest_pid = pid

    pmask = df["participant_id"] == closest_pid
    pdf = df[pmask].sort_values("date").reset_index(drop=True)
    z0 = Z[pmask.values][-1]

    env_cols = ["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"]
    beh_cols = ["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"]
    env_last = pdf[env_cols].tail(1).to_numpy(dtype=float, na_value=0.0)
    beh_last = pdf[beh_cols].tail(1).to_numpy(dtype=float, na_value=0.0)
    env_forecast = np.tile(env_last, (14, 1))
    beh_forecast = np.tile(beh_last, (14, 1))

    dynamics = LatentDynamicsModel(rng_seed=17)
    ptypes = available_perturbations()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    axes = axes.flat
    for ax, ptype in zip(axes, ptypes):
        spec = PerturbationSpec(perturbation_type=ptype, horizon_days=14)
        res = simulate_perturbation(z0, env_forecast, beh_forecast, spec, dynamics_model=dynamics)
        base = np.array(res.baseline_trajectory)
        cf = np.array(res.counterfactual_trajectory)
        diff = cf - base
        for i, name in enumerate(LATENT_DIM_NAMES):
            ax.plot(diff[:, i], label=name)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title(f"{ptype}\n(mag={res.magnitude})", fontsize=8)
        ax.set_xlabel("days")
    # Hide unused panel(s)
    for ax in list(axes)[len(ptypes):]:
        ax.axis("off")
    axes[0].set_ylabel("Δ latent (counterfactual − baseline)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    fig.suptitle(f"Counterfactual pathways for a typical participant ({closest_pid})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    out = fig_dir / "perturbation_pathways.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    processed_path = PROCESSED_DIR / "processed_features.csv"
    if not processed_path.exists():
        logger.error("Processed features missing. Run scripts/run_pipeline.py first.")
        return 1
    df = pd.read_csv(processed_path, parse_dates=["date"])
    if "sleep_duration_hours" not in df.columns:
        df = engineer_all_features(df)

    W, B, C, M, P = _modality_matrices(df)
    Z = encode_latent_states_classical(W, B, C, M, P).latent

    detector = fit_regime_detector(Z, n_clusters=4, random_state=17)
    regimes = detector.predict(Z)

    fig_dir = RESULTS_DIR / "figures"
    ensure_dir(fig_dir)

    outputs = []
    outputs.append(figure_latent_distribution(Z, fig_dir))
    outputs.append(figure_regime_phase_diagram(Z, regimes, fig_dir))
    outputs.append(figure_energy_landscape(Z, fig_dir))
    out_env = figure_environmental_response(df, Z, fig_dir)
    if out_env is not None:
        outputs.append(out_env)
    outputs.append(figure_perturbation_pathways(df, Z, fig_dir))

    print("\nGenerated figures:")
    for p in outputs:
        print(f"  {p}")
    print("\n(research prototype, non-clinical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
