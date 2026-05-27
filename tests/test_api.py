"""Tests for the FastAPI service.

Skipped automatically when FastAPI / Starlette TestClient are unavailable.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import app

    assert app is not None
    return TestClient(app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "disclaimer" in data
    assert "latent_dim_names" in data
    assert len(data["latent_dim_names"]) == 6


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _window():
    days = 14
    return {
        "participant_id": "TEST001",
        "days": days,
        "wearable": {
            "sleep_duration_hours": [7.0 + 0.1 * i for i in range(days)],
            "hrv_rmssd": [50 + 0.5 * i for i in range(days)],
            "resting_hr": [62.0 - 0.1 * i for i in range(days)],
            "daily_steps": [8000 + 50 * i for i in range(days)],
            "recovery_score": [70 + 0.2 * i for i in range(days)],
        },
        "behavior": {
            "screen_time_minutes": [240 - 2 * i for i in range(days)],
            "mobility_radius_km": [5 + 0.05 * i for i in range(days)],
            "location_entropy": [0.6 + 0.005 * i for i in range(days)],
            "phone_unlock_count": [80 - 0.5 * i for i in range(days)],
        },
        "environment": {
            "temperature_c": [25 + 0.3 * i for i in range(days)],
            "nighttime_temperature_c": [18 + 0.2 * i for i in range(days)],
            "aqi": [50 + i for i in range(days)],
            "heat_wave_flag": [0.0] * days,
        },
        "missingness": {
            "missing_wearable_flag": [0.0] * days,
            "missing_phone_flag": [0.0] * days,
            "missing_survey_flag": [0.0] * days,
        },
        "baseline": {
            "baseline_hrv": 50.0,
            "baseline_resting_hr": 65.0,
            "baseline_climate_vulnerability": 0.0,
            "baseline_resilience": 0.0,
        },
    }


def test_encode_state(client):
    r = client.post("/encode-state", json=_window())
    assert r.status_code == 200
    data = r.json()
    assert "latent_state" in data
    assert len(data["latent_state"]) == 6
    assert "disclaimer" in data


def test_simulate_perturbation(client):
    payload = {
        "participant_id": "TEST001",
        "recent_window": _window(),
        "perturbation_type": "cooling",
        "horizon_days": 14,
    }
    r = client.post("/simulate-perturbation", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "baseline_trajectory" in data
    assert "counterfactual_trajectory" in data
    assert "pathway_explanation" in data
    assert "disclaimer" in data


def test_detect_regime(client):
    payload = {"participant_id": "TEST001", "recent_window": _window()}
    r = client.post("/detect-regime", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "current_regime" in data
    assert "regime_probabilities" in data
    assert "early_warning" in data
    assert "disclaimer" in data
