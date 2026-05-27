"""Tests for the latent state encoder and regime detector."""

import numpy as np


def test_latent_shape(latent_states: np.ndarray, engineered_cohort):
    assert latent_states.shape == (len(engineered_cohort), 6)


def test_latent_state_names():
    from states.latent_state_encoder import LATENT_DIM_NAMES

    assert len(LATENT_DIM_NAMES) == 6
    assert LATENT_DIM_NAMES[0] == "autonomic_recovery"
    assert "stress_load" in LATENT_DIM_NAMES
    assert "missingness_pressure" in LATENT_DIM_NAMES


def test_regime_detector_assigns_labels(latent_states):
    from states.regime_detector import fit_regime_detector

    detector = fit_regime_detector(latent_states, n_clusters=4, random_state=17)
    labels = detector.predict(latent_states)
    assert len(labels) == len(latent_states)
    label_set = set(labels)
    assert {"stable", "stressed", "dysregulated", "recovery"} >= label_set
    assert len(label_set) >= 2  # at least two regimes populate


def test_regime_risk_summary_has_required_keys(latent_states):
    from states.regime_detector import fit_regime_detector

    detector = fit_regime_detector(latent_states, n_clusters=4, random_state=17)
    summary = detector.regime_risk_summary(latent_states[:1])
    for key in (
        "prob_stable",
        "prob_stressed",
        "prob_dysregulated",
        "prob_recovery",
        "dysregulation_risk",
        "recovery_probability",
    ):
        assert key in summary


def test_critical_transition_warning_outputs(latent_states):
    from states.early_warning import critical_transition_warning_score

    w = critical_transition_warning_score(latent_states[:60])
    for key in (
        "variance_signal",
        "autocorrelation_signal",
        "instability_index",
        "distance_to_dysregulated",
        "warning_score",
    ):
        assert key in w, f"Missing {key}"
    # warning_score may be a per-day array or a scalar
    ws = np.atleast_1d(w["warning_score"])
    assert np.all((ws >= 0.0) & (ws <= 1.0))


def test_energy_landscape_runs(latent_states):
    from states.energy_landscape import estimate_energy_landscape

    landscape = estimate_energy_landscape(latent_states[:200], grid_size=20)
    assert landscape.energy.shape == (20, 20)
    assert np.isfinite(landscape.energy).all()


def test_transition_matrix_returns_tuple(latent_states):
    """Regression for Bug 12: transition_matrix must return (matrix, regimes)
    so callers (dashboard, notebook 04) can label rows/cols of the matrix.
    """
    from states.regime_detector import fit_regime_detector, transition_matrix

    detector = fit_regime_detector(latent_states, n_clusters=4, random_state=17)
    labels = detector.predict(latent_states)
    result = transition_matrix(labels)
    # Must be unpackable as (matrix, regimes), not just a bare ndarray
    matrix, regimes = result
    assert matrix.shape == (4, 4)
    assert len(regimes) == 4
    assert set(regimes) == {"stable", "stressed", "dysregulated", "recovery"}
    # Each row should be a probability distribution (sum to 1 or 0 if empty)
    for row in matrix:
        s = row.sum()
        assert abs(s - 1.0) < 1e-6 or s == 0.0


def test_critical_transition_warning_with_centroid(latent_states):
    """Regression for Bug 14: the warning function returns
    distance_to_dysregulated only when called with a centroid. Used to be
    called without one in the API and dashboard, dropping a signal."""
    import numpy as np
    from states.regime_detector import fit_regime_detector
    from states.early_warning import critical_transition_warning_score

    # Build a longer trajectory for stable detector fit
    Z = np.tile(latent_states, (3, 1))
    detector = fit_regime_detector(Z, n_clusters=4, random_state=17)
    dys_idx = detector.labels_to_regime.index("dysregulated")
    dys_centroid = detector.centroids[dys_idx]
    warning = critical_transition_warning_score(Z, dysregulated_centroid=dys_centroid)
    assert warning["distance_to_dysregulated"] is not None
    assert np.isfinite(np.asarray(warning["distance_to_dysregulated"])).all()


def test_api_last_finite_helper():
    """Helper used by /detect-regime to surface scalar warning values."""
    import numpy as np
    from api.main import _last_finite

    assert _last_finite([1.0, 2.0, 3.0]) == 3.0
    assert _last_finite([np.nan, np.nan, 0.5]) == 0.5
    assert _last_finite([np.nan, np.nan]) == 0.0
    assert _last_finite([]) == 0.0
