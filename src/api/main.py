"""FastAPI service for the latent human dynamics engine.

The service exposes four endpoints: a description, a health probe, latent
state encoding, perturbation simulation, and regime detection. Every
non-trivial response carries the standard non-clinical disclaimer and is
passed through ``validate_safe_output`` before being returned.

Run with::

    uvicorn src.api.main:app --reload

If FastAPI is not installed the module still imports cleanly; the ``app``
attribute will be ``None`` and ``create_app()`` will raise a clear error.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure src/ is importable when the module is run directly via uvicorn from
# the repo root (``uvicorn src.api.main:app``).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from safety.clinical_guardrails import validate_safe_output  # noqa: E402
from safety.output_disclaimer import DISCLAIMER  # noqa: E402
from states.latent_state_encoder import (  # noqa: E402
    LATENT_DIM_NAMES,
    encode_latent_states_classical,
)
from states.regime_detector import fit_regime_detector  # noqa: E402
from states.early_warning import critical_transition_warning_score  # noqa: E402
from dynamics.transition_model import LatentDynamicsModel  # noqa: E402
from counterfactuals.perturbation_engine import (  # noqa: E402
    PerturbationSpec,
    simulate_perturbation,
)

try:
    from fastapi import FastAPI, HTTPException

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers shared by all endpoints
# ---------------------------------------------------------------------------


def _stack_modality(payload: Dict[str, Any], channels: list[str], default_T: int) -> np.ndarray:
    """Stack a dict-of-lists into a (T, C) numpy array with safe defaults."""
    columns = []
    T = default_T
    for ch in channels:
        values = payload.get(ch)
        if values is None:
            columns.append(np.zeros(T))
        else:
            arr = np.array([0.0 if v is None else float(v) for v in values], dtype=float)
            T = max(T, len(arr))
            columns.append(arr)
    # Pad to common length
    columns = [np.pad(c, (0, T - len(c)), constant_values=0.0) for c in columns]
    return np.stack(columns, axis=1)


def _encode_recent_window(window: Dict[str, Any]) -> np.ndarray:
    """Reduce a recent window payload to a single latent state vector.

    Uses the classical (PCA-based) encoder so the API works without a trained
    neural model checkpoint. The encoder is fit on the provided window — this
    is a degenerate fit for a single short window, but it produces a deterministic
    vector usable for downstream calls. In production this would be replaced
    by a trained ``MultimodalLatentStateEncoder`` checkpoint.
    """
    T = int(window.get("days", 14))
    W = _stack_modality(
        window.get("wearable", {}),
        ["sleep_duration_hours", "hrv_rmssd", "resting_hr", "daily_steps", "recovery_score"],
        T,
    )
    B = _stack_modality(
        window.get("behavior", {}),
        ["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"],
        T,
    )
    C = _stack_modality(
        window.get("environment", {}),
        ["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"],
        T,
    )
    M = _stack_modality(
        window.get("missingness", {}),
        ["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"],
        T,
    )
    baseline = window.get("baseline", {}) or {}
    P = np.tile(
        np.array(
            [
                baseline.get("baseline_hrv", 50.0),
                baseline.get("baseline_resting_hr", 65.0),
                baseline.get("baseline_climate_vulnerability", 0.0),
                baseline.get("baseline_resilience", 0.0),
            ],
            dtype=float,
        ),
        (T, 1),
    )

    # We need at least a handful of timesteps for PCA. If the window is short,
    # tile it so the classical encoder produces a meaningful projection.
    if T < 12:
        reps = int(np.ceil(12 / max(T, 1)))
        W = np.tile(W, (reps, 1))
        B = np.tile(B, (reps, 1))
        C = np.tile(C, (reps, 1))
        M = np.tile(M, (reps, 1))
        P = np.tile(P, (reps, 1))

    result = encode_latent_states_classical(W, B, C, M, P)
    # Return the most recent day as the participant's "current" state.
    # (PCA centers each axis to mean 0, so averaging over the window collapses
    # to zero; the final day preserves end-of-window position in latent space.)
    return result.latent[-1]


def _last_finite(arr) -> float:
    """Return the most recent finite value from an array-like, or 0.0."""
    a = np.atleast_1d(np.asarray(arr, dtype=float))
    finite = a[np.isfinite(a)]
    return float(finite[-1]) if finite.size > 0 else 0.0


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app():
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Install with `pip install fastapi uvicorn` "
            "or use `pip install -r requirements.txt`."
        )

    app = FastAPI(
        title="Latent Human Dynamics Lab",
        description=(
            "Research API for the multimodal latent state-space engine. "
            "Non-clinical, non-diagnostic. See the disclaimer on every response."
        ),
        version="0.1.0",
    )

    @app.get("/")
    def root() -> Dict[str, Any]:
        return validate_safe_output(
            {
                "name": "latent-human-dynamics-lab",
                "description": (
                    "Multimodal human state-space engine for modeling "
                    "physiological, behavioral, and environmental dynamics."
                ),
                "latent_dim_names": list(LATENT_DIM_NAMES),
                "endpoints": ["/", "/health", "/encode-state", "/simulate-perturbation", "/detect-regime"],
                "disclaimer": DISCLAIMER,
            }
        )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "disclaimer": DISCLAIMER}

    @app.post("/encode-state")
    def encode_state(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            window = payload if "wearable" in payload else payload.get("recent_window", payload)
            z = _encode_recent_window(window)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=f"Could not encode state: {exc}")
        return validate_safe_output(
            {
                "participant_id": payload.get("participant_id", "unknown"),
                "latent_state": z.tolist(),
                "latent_dim_names": list(LATENT_DIM_NAMES),
                "uncertainty": [0.1] * len(z),
                "disclaimer": DISCLAIMER,
            }
        )

    @app.post("/simulate-perturbation")
    def simulate_perturbation_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            window = payload.get("recent_window", {})
            z0 = _encode_recent_window(window)

            horizon = int(payload.get("horizon_days", 14))
            T = int(window.get("days", 14))
            env = _stack_modality(
                window.get("environment", {}),
                ["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"],
                T,
            )
            beh = _stack_modality(
                window.get("behavior", {}),
                ["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"],
                T,
            )
            # Repeat last day as constant forecast
            env_forecast = np.tile(env[-1:], (horizon, 1)) if len(env) else np.zeros((horizon, 4))
            beh_forecast = np.tile(beh[-1:], (horizon, 1)) if len(beh) else np.zeros((horizon, 4))

            spec = PerturbationSpec(
                perturbation_type=payload["perturbation_type"],
                magnitude=payload.get("magnitude"),
                horizon_days=horizon,
            )
            dynamics = LatentDynamicsModel()
            result = simulate_perturbation(
                z0,
                env_forecast,
                beh_forecast,
                spec,
                dynamics_model=dynamics,
            )
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"Missing field: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Simulation failed: {exc}")

        return validate_safe_output(result.to_dict())

    @app.post("/detect-regime")
    def detect_regime(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            window = payload.get("recent_window", {})
            T = int(window.get("days", 14))
            W = _stack_modality(
                window.get("wearable", {}),
                ["sleep_duration_hours", "hrv_rmssd", "resting_hr", "daily_steps", "recovery_score"],
                T,
            )
            B = _stack_modality(
                window.get("behavior", {}),
                ["screen_time_minutes", "mobility_radius_km", "location_entropy", "phone_unlock_count"],
                T,
            )
            C = _stack_modality(
                window.get("environment", {}),
                ["temperature_c", "nighttime_temperature_c", "aqi", "heat_wave_flag"],
                T,
            )
            M = _stack_modality(
                window.get("missingness", {}),
                ["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"],
                T,
            )
            baseline = window.get("baseline", {}) or {}
            P = np.tile(
                np.array(
                    [
                        baseline.get("baseline_hrv", 50.0),
                        baseline.get("baseline_resting_hr", 65.0),
                        baseline.get("baseline_climate_vulnerability", 0.0),
                        baseline.get("baseline_resilience", 0.0),
                    ]
                ),
                (max(T, 12), 1),
            )
            if T < 12:
                reps = int(np.ceil(12 / max(T, 1)))
                W = np.tile(W, (reps, 1))
                B = np.tile(B, (reps, 1))
                C = np.tile(C, (reps, 1))
                M = np.tile(M, (reps, 1))

            Z = encode_latent_states_classical(W, B, C, M, P).latent
            detector = fit_regime_detector(Z, n_clusters=4, random_state=17)
            current = detector.predict(Z[-1:])[0]
            risk_summary = detector.regime_risk_summary(Z[-1:])
            # Pass the dysregulated centroid so the distance signal is
            # populated. Without this, distance_to_dysregulated is None
            # and the downstream summary silently drops it.
            try:
                dys_idx = detector.labels_to_regime.index("dysregulated")
                dys_centroid = detector.centroids[dys_idx]
            except (ValueError, AttributeError):
                dys_centroid = None
            warning = critical_transition_warning_score(
                Z, dysregulated_centroid=dys_centroid
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Regime detection failed: {exc}")

        return validate_safe_output(
            {
                "participant_id": payload.get("participant_id", "unknown"),
                "current_regime": current,
                "regime_probabilities": {
                    "stable": float(risk_summary.get("prob_stable", 0.0)),
                    "stressed": float(risk_summary.get("prob_stressed", 0.0)),
                    "dysregulated": float(risk_summary.get("prob_dysregulated", 0.0)),
                    "recovery": float(risk_summary.get("prob_recovery", 0.0)),
                },
                "dysregulation_risk": float(risk_summary.get("dysregulation_risk", 0.0)),
                "recovery_probability": float(risk_summary.get("recovery_probability", 0.0)),
                "early_warning": {
                    # The warning function returns per-day arrays. We surface
                    # the most recent finite value as the API's "current"
                    # warning level. Early days of a trajectory often produce
                    # NaN because rolling windows can't fill, so we fall back
                    # to the last non-NaN value (or 0.0 if everything is NaN).
                    k: _last_finite(v)
                    for k, v in warning.items()
                    if v is not None
                },
                "disclaimer": DISCLAIMER,
            }
        )

    return app


# Module-level app for `uvicorn src.api.main:app`
app = create_app() if _HAS_FASTAPI else None
