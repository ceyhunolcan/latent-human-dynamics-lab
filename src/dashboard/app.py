"""Streamlit dashboard for the Latent Human Dynamics Lab.

This is a research visualization layer over the synthetic cohort and the
latent / dynamics / counterfactual stack. Every page surfaces the standard
non-clinical research disclaimer. Plotting is matplotlib only.

Run with::

    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make src/ importable regardless of how Streamlit launches the file
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import streamlit as st
    import matplotlib.pyplot as plt

    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False
    st = None  # type: ignore[assignment]


from safety.output_disclaimer import DISCLAIMER  # noqa: E402
from utils.paths import SYNTHETIC_DIR, PROCESSED_DIR  # noqa: E402
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
from states.early_warning import critical_transition_warning_score  # noqa: E402
from dynamics.transition_model import LatentDynamicsModel  # noqa: E402
from dynamics.resilience_model import estimate_resilience_profile  # noqa: E402
from counterfactuals.perturbation_engine import (  # noqa: E402
    PerturbationSpec,
    available_perturbations,
    simulate_perturbation,
)
from counterfactuals.intervention_simulator import summarise_intervention_pathway  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


def _load_cohort() -> pd.DataFrame:
    processed = PROCESSED_DIR / "processed_features.csv"
    synthetic = SYNTHETIC_DIR / "synthetic_cohort.csv"
    if processed.exists():
        return pd.read_csv(processed, parse_dates=["date"])
    if synthetic.exists():
        df = pd.read_csv(synthetic, parse_dates=["date"])
        return engineer_all_features(df)
    raise FileNotFoundError(
        "No cohort data found. Run `python scripts/run_pipeline.py` first."
    )


def _modality_matrices(df: pd.DataFrame):
    """Pull the modality channels the classical encoder expects."""

    def pick(cols, default=0.0):
        present = [c for c in cols if c in df.columns]
        if not present:
            return np.zeros((len(df), len(cols)))
        mat = df[present].to_numpy(dtype=float, na_value=default)
        if mat.shape[1] < len(cols):
            pad = np.zeros((len(df), len(cols) - mat.shape[1]))
            mat = np.concatenate([mat, pad], axis=1)
        return mat

    W = pick(["sleep_duration_hours", "hrv_rmssd", "resting_hr", "daily_steps", "recovery_score"])
    B = pick(["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"])
    C = pick(["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"])
    M = pick(["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"])
    P_cols = [
        "baseline_hrv",
        "baseline_resting_hr",
        "baseline_climate_vulnerability",
        "baseline_resilience",
    ]
    P = pick(P_cols)
    return W, B, C, M, P


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def run() -> None:
    if not _HAS_STREAMLIT:
        raise RuntimeError(
            "Streamlit is not installed. Run `pip install streamlit` "
            "(or `pip install -r requirements.txt`) to launch the dashboard."
        )

    st.set_page_config(
        page_title="Latent Human Dynamics Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Latent Human Dynamics Lab")
    st.caption(
        "A multimodal human state-space engine for modeling physiological, "
        "behavioral, and environmental dynamics."
    )
    st.info(DISCLAIMER)

    try:
        cohort = _load_cohort()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    participants = sorted(cohort["participant_id"].unique().tolist())
    section = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "Participant Latent Trajectory",
            "State Space Map",
            "Energy Landscape",
            "Regime Transitions",
            "Environmental Forcing",
            "Counterfactual Perturbation Simulator",
            "Resilience Profile",
            "Missingness Dynamics",
            "Model Evaluation",
            "Safety Notice",
        ],
    )
    pid = st.sidebar.selectbox("Participant", participants)

    pdf = cohort[cohort["participant_id"] == pid].sort_values("date").reset_index(drop=True)

    # Pre-compute latent states for all participants (cheap on synthetic scale)
    W, B, C, M, P = _modality_matrices(cohort)
    Z = encode_latent_states_classical(W, B, C, M, P).latent
    cohort_with_z = cohort.copy().reset_index(drop=True)
    for i, name in enumerate(LATENT_DIM_NAMES):
        cohort_with_z[f"z_{name}"] = Z[:, i]
    pZ = cohort_with_z[cohort_with_z["participant_id"] == pid][
        [f"z_{n}" for n in LATENT_DIM_NAMES]
    ].to_numpy()

    detector = fit_regime_detector(Z, n_clusters=4, random_state=17)
    regime_labels = detector.predict(Z)
    cohort_with_z["regime"] = regime_labels
    p_regimes = cohort_with_z[cohort_with_z["participant_id"] == pid]["regime"].tolist()

    if section == "Overview":
        st.subheader("Cohort overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Participants", cohort["participant_id"].nunique())
        c2.metric("Days observed", cohort["date"].nunique())
        c3.metric("Total observations", len(cohort))
        c4.metric("Latent dimensions", len(LATENT_DIM_NAMES))
        st.markdown(
            "This dashboard visualises a synthetic cohort generated by the lab's "
            "state-space simulator. The latent states are computed by a classical "
            "PCA-based encoder when no trained checkpoint is available."
        )
        st.markdown("**Regime distribution across cohort:**")
        regime_counts = cohort_with_z["regime"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(regime_counts.index, regime_counts.values)
        ax.set_ylabel("Person-days")
        ax.set_xlabel("Regime")
        st.pyplot(fig)

    elif section == "Participant Latent Trajectory":
        st.subheader(f"Latent trajectory — {pid}")
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(LATENT_DIM_NAMES):
            ax.plot(pdf["date"], pZ[:, i], label=name, alpha=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Latent value (z-score)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_title("Latent state evolution")
        st.pyplot(fig)
        st.markdown(
            "Each line is one inferred latent dimension. Smooth multi-day drifts "
            "reflect the dynamics model; sudden jumps usually indicate regime shifts "
            "or environmental shocks."
        )

    elif section == "State Space Map":
        st.subheader("Latent state space (2D PCA projection)")
        Z2 = project_latent_2d(Z)
        fig, ax = plt.subplots(figsize=(8, 6))
        # Background cloud
        ax.scatter(Z2[:, 0], Z2[:, 1], c="lightgrey", s=4, alpha=0.4, label="cohort")
        # Trajectory for selected participant
        pZ2 = project_latent_2d(pZ)
        ax.plot(pZ2[:, 0], pZ2[:, 1], "-", linewidth=1.5, alpha=0.8)
        ax.scatter(pZ2[0, 0], pZ2[0, 1], c="green", s=80, marker="o", label="start", zorder=5)
        ax.scatter(pZ2[-1, 0], pZ2[-1, 1], c="red", s=80, marker="X", label="end", zorder=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"State space trajectory — {pid}")
        ax.legend()
        st.pyplot(fig)

    elif section == "Energy Landscape":
        st.subheader("Pseudo-energy landscape  E(z) = -log p(z)")
        landscape = estimate_energy_landscape(Z, grid_size=60)
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_energy_landscape(landscape, ax=ax)
        pZ2 = project_latent_2d(pZ)
        ax.plot(pZ2[:, 0], pZ2[:, 1], color="white", linewidth=1.2, alpha=0.9)
        ax.set_title(f"Energy landscape with {pid}'s trajectory overlaid")
        st.pyplot(fig)
        st.markdown(
            "Darker basins correspond to states the cohort occupies often; ridges "
            "are rare or unstable configurations. This is a descriptive visualisation, "
            "not a clinical readout."
        )

    elif section == "Regime Transitions":
        st.subheader(f"Regime timeline — {pid}")
        regime_colors = {
            "stable": "#4daf4a",
            "stressed": "#ff7f00",
            "dysregulated": "#e41a1c",
            "recovery": "#377eb8",
        }
        fig, ax = plt.subplots(figsize=(10, 2.5))
        for i, r in enumerate(p_regimes):
            ax.axvspan(i - 0.5, i + 0.5, color=regime_colors.get(r, "grey"), alpha=0.7)
        ax.set_xlim(0, len(p_regimes))
        ax.set_yticks([])
        ax.set_xlabel("Day index")
        ax.set_title("Regime sequence")
        st.pyplot(fig)

        st.markdown("**Transition matrix (cohort-wide):**")
        from states.regime_detector import transition_matrix

        T_mat, regimes = transition_matrix(regime_labels)
        st.dataframe(pd.DataFrame(T_mat, index=regimes, columns=regimes).round(3))

        # Try to extract the dysregulated centroid so distance_to_dysregulated
        # is populated. Fall back to None if the detector isn't available
        # (e.g. on an empty regime set).
        try:
            dys_idx = detector.labels_to_regime.index("dysregulated")
            dys_centroid = detector.centroids[dys_idx]
        except (ValueError, AttributeError, NameError):
            dys_centroid = None
        warning = critical_transition_warning_score(pZ, dysregulated_centroid=dys_centroid)

        def _last(arr):
            """Most recent finite value, or NaN if everything is missing."""
            if arr is None:
                return float("nan")
            a = np.atleast_1d(np.asarray(arr, dtype=float))
            finite = a[np.isfinite(a)]
            return float(finite[-1]) if finite.size > 0 else float("nan")

        st.markdown(
            f"**Critical transition warning score:** {_last(warning['warning_score']):.3f}  \n"
            f"variance signal: {_last(warning['variance_signal']):.3f} · "
            f"autocorrelation: {_last(warning['autocorrelation_signal']):.3f} · "
            f"instability: {_last(warning['instability_index']):.3f} · "
            f"distance to dysregulated: {_last(warning['distance_to_dysregulated']):.3f}"
        )

    elif section == "Environmental Forcing":
        st.subheader(f"Environmental forcing — {pid}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        for ax, col in zip(axes.flat, ["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"]):
            if col in pdf.columns:
                ax.plot(pdf["date"], pdf[col])
                ax.set_title(col)
        fig.tight_layout()
        st.pyplot(fig)
        if "environmental_physiological_load" in pdf.columns:
            st.markdown(
                f"**Mean Environmental Physiological Load (EPL):** "
                f"{pdf['environmental_physiological_load'].mean():.3f}"
            )

    elif section == "Counterfactual Perturbation Simulator":
        st.subheader(f"Counterfactual perturbation — {pid}")
        ptype = st.selectbox("Perturbation type", available_perturbations())
        horizon = st.slider("Horizon (days)", 7, 30, 14)
        magnitude = st.text_input(
            "Magnitude (blank = default)",
            value="",
            help="e.g. 45 for sleep_extension (minutes), -4 for cooling (°C)",
        )
        mag = float(magnitude) if magnitude.strip() else None

        if st.button("Simulate"):
            T = min(14, len(pZ))
            z0 = pZ[-1] if len(pZ) else np.zeros(6)
            env_cols = ["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"]
            beh_cols = [
                "screen_time_minutes",
                "mobility_radius_km",
                "location_entropy",
                "phone_unlock_count",
            ]
            env_last = (
                pdf[env_cols].tail(1).to_numpy(dtype=float, na_value=0.0)
                if all(c in pdf.columns for c in env_cols)
                else np.zeros((1, 4))
            )
            beh_last = (
                pdf[beh_cols].tail(1).to_numpy(dtype=float, na_value=0.0)
                if all(c in pdf.columns for c in beh_cols)
                else np.zeros((1, 4))
            )
            env_forecast = np.tile(env_last, (horizon, 1))
            beh_forecast = np.tile(beh_last, (horizon, 1))

            spec = PerturbationSpec(perturbation_type=ptype, magnitude=mag, horizon_days=horizon)
            dynamics = LatentDynamicsModel()
            result = simulate_perturbation(z0, env_forecast, beh_forecast, spec, dynamics_model=dynamics)

            st.markdown(summarise_intervention_pathway(result))

            fig, ax = plt.subplots(figsize=(10, 5))
            base = np.array(result.baseline_trajectory)
            cf = np.array(result.counterfactual_trajectory)
            t = np.arange(len(base))
            for i, name in enumerate(LATENT_DIM_NAMES):
                ax.plot(t, base[:, i], "--", color=f"C{i}", alpha=0.5)
                ax.plot(t, cf[:, i], "-", color=f"C{i}", label=name)
            ax.set_xlabel("Days from intervention")
            ax.set_ylabel("Latent value")
            ax.set_title(f"Baseline (dashed) vs counterfactual (solid) — {ptype}")
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            st.pyplot(fig)

    elif section == "Resilience Profile":
        st.subheader(f"Resilience profile — {pid}")
        profile = estimate_resilience_profile(pdf, participant_id=pid)
        st.json(profile.to_dict())
        st.markdown(
            "**Recovery half-life** is the number of days for stress-load excursions "
            "to decay by 50% after a perturbation. **Environmental sensitivity** captures "
            "how strongly heat and AQI propagate into physiological state."
        )

    elif section == "Missingness Dynamics":
        st.subheader(f"Missingness over time — {pid}")
        miss_cols = [
            c for c in ["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"]
            if c in pdf.columns
        ]
        if miss_cols:
            fig, ax = plt.subplots(figsize=(10, 3))
            for c in miss_cols:
                ax.plot(pdf["date"], pdf[c].rolling(7, min_periods=1).mean(), label=c)
            ax.set_ylabel("7-day rolling rate")
            ax.set_xlabel("Date")
            ax.legend()
            st.pyplot(fig)
        if "missingness_pressure" in pdf.columns:
            st.markdown(
                f"**Mean missingness pressure proxy:** {pdf['missingness_pressure'].mean():.3f}"
            )
        st.markdown(
            "Missingness in passive sensing is state-dependent: people drop out more "
            "when stressed or unwell. Treating it as random understates risk in the "
            "very subgroups the system is supposed to support."
        )

    elif section == "Model Evaluation":
        st.subheader("Model leaderboard")
        leaderboard_path = _REPO_ROOT / "results" / "model_leaderboard.csv"
        if leaderboard_path.exists():
            st.dataframe(pd.read_csv(leaderboard_path))
        else:
            st.warning("No leaderboard found yet. Run training scripts to populate.")

    elif section == "Safety Notice":
        st.subheader("Non-clinical research notice")
        st.warning(DISCLAIMER)
        st.markdown(
            "This system is a research prototype for **modeling and simulation** of "
            "human physiological and behavioral dynamics. It does **not** provide "
            "medical advice, diagnosis, or treatment, and it has not been validated "
            "against clinical outcomes. Outputs labelled as *risk* or *probability* "
            "refer to **research signals over simulated trajectories**, not clinical "
            "events. No regulatory clearance has been sought or obtained for any use."
        )


if __name__ == "__main__":
    if _HAS_STREAMLIT:
        run()
    else:
        print(
            "Streamlit is not installed. Run `pip install streamlit` "
            "and then `streamlit run src/dashboard/app.py`."
        )
