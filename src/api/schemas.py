"""Request and response schemas for the FastAPI service.

We import pydantic lazily so the module is still importable in environments
where pydantic is not installed (e.g. when only running the synthetic data
pipeline). When pydantic is unavailable we fall back to lightweight
dataclasses that expose enough of the same surface to be useful in tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_PYDANTIC = False

    from dataclasses import dataclass, field

    def Field(default=None, **_kwargs):  # type: ignore[no-redef]
        return default

    @dataclass
    class BaseModel:  # type: ignore[no-redef]
        """Minimal dataclass stand-in for pydantic.BaseModel."""

        def dict(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Encode-state
# ---------------------------------------------------------------------------


class EncodeStateRequest(BaseModel):
    """Recent participant window used to encode a latent state vector.

    Each list is expected to have the same length T (number of days). Missing
    modality entries can be passed as ``null`` and the encoder will treat them
    as missing rather than zero.
    """

    participant_id: str = Field(..., description="Opaque participant identifier")
    days: int = Field(14, description="Length of the window in days")
    wearable: Dict[str, List[Optional[float]]] = Field(
        default_factory=dict,
        description="Wearable channels: sleep_duration_hours, hrv_rmssd, resting_hr, ...",
    )
    behavior: Dict[str, List[Optional[float]]] = Field(default_factory=dict)
    environment: Dict[str, List[Optional[float]]] = Field(default_factory=dict)
    missingness: Dict[str, List[Optional[float]]] = Field(default_factory=dict)
    baseline: Dict[str, float] = Field(
        default_factory=dict,
        description="Participant-level baselines (age, baseline_hrv, ...)",
    )


class LatentStateResponse(BaseModel):
    participant_id: str
    latent_state: List[float]
    latent_dim_names: List[str]
    uncertainty: List[float]
    disclaimer: str


# ---------------------------------------------------------------------------
# Simulate-perturbation
# ---------------------------------------------------------------------------


class SimulatePerturbationRequest(BaseModel):
    participant_id: str
    recent_window: EncodeStateRequest
    perturbation_type: str = Field(
        ...,
        description=(
            "One of: sleep_extension, screen_reduction, exercise_boost, "
            "cooling, air_quality_improvement, heat_wave_shock, "
            "combined_resilience_protocol"
        ),
    )
    magnitude: Optional[float] = Field(
        None,
        description="Magnitude in natural units. If null, the engine uses a default.",
    )
    horizon_days: int = Field(14, ge=1, le=60)


class SimulatePerturbationResponse(BaseModel):
    participant_id: str
    perturbation_type: str
    magnitude: float
    horizon_days: int
    baseline_trajectory: List[List[float]]
    counterfactual_trajectory: List[List[float]]
    latent_state_delta: List[float]
    observed_proxy_delta: Dict[str, float]
    uncertainty: Optional[Dict[str, List[float]]]
    pathway_explanation: str
    disclaimer: str


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


class DetectRegimeRequest(BaseModel):
    participant_id: str
    recent_window: EncodeStateRequest


class DetectRegimeResponse(BaseModel):
    participant_id: str
    current_regime: str
    regime_probabilities: Dict[str, float]
    dysregulation_risk: float
    recovery_probability: float
    early_warning: Dict[str, float]
    disclaimer: str


__all__ = [
    "EncodeStateRequest",
    "LatentStateResponse",
    "SimulatePerturbationRequest",
    "SimulatePerturbationResponse",
    "DetectRegimeRequest",
    "DetectRegimeResponse",
]
