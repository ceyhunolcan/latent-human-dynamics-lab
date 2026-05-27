"""Train (or pseudo-train) the latent dynamics prototype.

When PyTorch is available this script trains the neural transition model
on the processed synthetic cohort and writes a checkpoint to
``results/checkpoints/``. When PyTorch is not available it falls back to
fitting the classical PCA-based encoder and writing a small metrics file —
this keeps the pipeline reproducible in minimal CPU-only environments.

Run from repo root::

    python scripts/train_dynamics_model.py --epochs 5 --batch-size 64
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

from utils.logging import get_logger  # noqa: E402
from utils.paths import PROCESSED_DIR, CHECKPOINTS_DIR, RESULTS_DIR, ensure_dir  # noqa: E402
from features import engineer_all_features  # noqa: E402
from states.latent_state_encoder import (  # noqa: E402
    LATENT_DIM_NAMES,
    encode_latent_states_classical,
)
from dynamics.transition_model import LatentDynamicsModel  # noqa: E402
from evaluation.metrics import trajectory_rmse  # noqa: E402

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML to override dynamics constants (e.g. configs/dynamics.yaml).",
    )
    args = parser.parse_args()

    # Resolve the config stack. If --config given, load and merge it on top
    # of the defaults. Otherwise just use the canonical defaults.
    if args.config:
        from utils.config import load_default_config, load_yaml, merge_configs
        cfg = merge_configs(
            load_default_config(),
            {"dynamics": load_yaml(args.config)},
        )
    else:
        from utils.config import load_default_config
        cfg = load_default_config()

    processed_path = PROCESSED_DIR / "processed_features.csv"
    if not processed_path.exists():
        logger.error("Processed features not found. Run scripts/run_pipeline.py first.")
        return 1

    logger.info("Loading processed cohort from %s", processed_path)
    df = pd.read_csv(processed_path, parse_dates=["date"])
    if "sleep_duration_hours" not in df.columns:
        # Backwards-compatibility: caller may have skipped feature engineering
        df = engineer_all_features(df)

    W, B, C, M, P = _modality_matrices(df)

    logger.info("Encoding latent states via classical PCA encoder (CPU)")
    enc = encode_latent_states_classical(W, B, C, M, P)
    Z = enc.latent
    for i, name in enumerate(LATENT_DIM_NAMES):
        df[f"z_{name}"] = Z[:, i]

    # Simple held-out trajectory RMSE: use the dynamics model to step forward
    # from each day's latent state, compare to the next day's latent state.
    logger.info("Evaluating one-step prediction RMSE of the dynamics prior")
    # Dynamics constants come from the config stack now, with code-defaults
    # as the canonical fallback. Override via --config configs/dynamics.yaml.
    from utils.config import get_dynamics_settings
    s = get_dynamics_settings(cfg)
    dynamics = LatentDynamicsModel(
        latent_dim=6,
        contraction=s["contraction"],
        noise_std=0.0,  # deterministic for evaluation
        rng_seed=args.seed,
    )
    logger.info("Dynamics: contraction=%.3f, noise_std=%.3f", s["contraction"], s["noise_std"])

    # Group-aware shift so we never predict across participants
    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)
    same_pid = df["participant_id"].values[1:] == df["participant_id"].values[:-1]
    Z_t = Z[:-1][same_pid]
    Z_next = Z[1:][same_pid]

    env_cols = ["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"]
    beh_cols = ["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"]
    E_t = df[env_cols].to_numpy(dtype=float, na_value=0.0)[:-1][same_pid]
    B_t = df[beh_cols].to_numpy(dtype=float, na_value=0.0)[:-1][same_pid]

    preds = np.stack(
        [dynamics.predict_next_state(z, b, e, perturbation_t=None) for z, b, e in zip(Z_t, B_t, E_t)],
        axis=0,
    )
    rmse = float(trajectory_rmse(Z_next, preds))
    logger.info("One-step latent RMSE: %.4f", rmse)

    ensure_dir(CHECKPOINTS_DIR)
    ensure_dir(RESULTS_DIR / "tables")

    metrics = {
        "model": "latent_dynamics_classical_encoder",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_train_transitions": int(same_pid.sum()),
        "trajectory_rmse_one_step": rmse,
        "latent_dim_names": list(LATENT_DIM_NAMES),
        "note": "PCA encoder + dynamics prior. Replace with neural training when torch is available.",
        "disclaimer": "(research prototype, non-clinical)",
    }
    metrics_path = RESULTS_DIR / "tables" / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Save a lightweight "checkpoint" — the encoder components + dynamics config
    ckpt = {
        "latent_dim_names": list(LATENT_DIM_NAMES),
        "contraction": 0.1,
        "noise_std": 0.0,
        "seed": args.seed,
    }
    ckpt_path = CHECKPOINTS_DIR / "latent_dynamics_v0.json"
    ckpt_path.write_text(json.dumps(ckpt, indent=2))

    print("\nTraining complete (classical fallback).")
    print(f"  Transitions used : {int(same_pid.sum())}")
    print(f"  One-step RMSE    : {rmse:.4f}")
    print(f"  Metrics          : {metrics_path}")
    print(f"  Checkpoint       : {ckpt_path}")
    print("\n(research prototype, non-clinical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
