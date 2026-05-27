"""Tests for the counterfactual perturbation engine."""

import numpy as np


def test_available_perturbations_set():
    from counterfactuals.perturbation_engine import available_perturbations

    perts = available_perturbations()
    assert "sleep_extension" in perts
    assert "cooling" in perts
    assert "heat_wave_shock" in perts
    assert "combined_resilience_protocol" in perts


def test_simulate_perturbation_returns_disclaimer():
    from counterfactuals.perturbation_engine import (
        PerturbationSpec,
        simulate_perturbation,
    )
    from dynamics.transition_model import LatentDynamicsModel

    z0 = np.zeros(6)
    env = np.zeros((14, 4))
    beh = np.zeros((14, 4))
    spec = PerturbationSpec(perturbation_type="sleep_extension", horizon_days=14)
    res = simulate_perturbation(
        z0, env, beh, spec, dynamics_model=LatentDynamicsModel(rng_seed=17)
    )
    d = res.to_dict()
    assert "disclaimer" in d
    assert "medical" in d["disclaimer"].lower()
    assert len(d["baseline_trajectory"]) == 14  # one entry per horizon day


def test_compare_trajectories_proxy_deltas():
    from counterfactuals.trajectory_comparator import compare_trajectories

    base = np.zeros((15, 6))
    cf = np.zeros((15, 6))
    cf[:, 2] = -0.3  # reduce stress_load
    comp = compare_trajectories(base, cf)
    assert "stress_score" in comp.observed_proxy_delta


def test_summarise_intervention_pathway_contains_disclaimer():
    from counterfactuals.intervention_simulator import summarise_intervention_pathway
    from counterfactuals.perturbation_engine import (
        PerturbationSpec,
        simulate_perturbation,
    )
    from dynamics.transition_model import LatentDynamicsModel

    z0 = np.zeros(6)
    env = np.zeros((14, 4))
    beh = np.zeros((14, 4))
    spec = PerturbationSpec(perturbation_type="cooling", horizon_days=14)
    res = simulate_perturbation(
        z0, env, beh, spec, dynamics_model=LatentDynamicsModel(rng_seed=17)
    )
    summary = summarise_intervention_pathway(res)
    assert "Research prototype" in summary or "medical" in summary.lower()


def test_unknown_perturbation_type_lists_valid_options():
    """Regression for Bug 16: unknown perturbation_type error should list
    valid alternatives so users can fix their typo."""
    import numpy as np
    from counterfactuals.perturbation_engine import PerturbationSpec, simulate_perturbation
    from dynamics.transition_model import LatentDynamicsModel

    try:
        simulate_perturbation(
            np.zeros(6), np.zeros((14, 4)), np.zeros((14, 4)),
            PerturbationSpec(perturbation_type="not_a_real_type", horizon_days=14),
            dynamics_model=LatentDynamicsModel(),
        )
    except ValueError as e:
        msg = str(e)
        # Must contain at least one valid alternative
        assert "sleep_extension" in msg or "cooling" in msg, (
            f"error message must list valid types; got: {msg}"
        )
        # Must contain the bad name
        assert "not_a_real_type" in msg


def test_perturbation_magnitude_nan_rejected():
    """Regression for Bug 17: NaN/inf magnitudes used to silently propagate
    through the dynamics, producing all-NaN counterfactual trajectories.
    They should now be rejected with a clear error."""
    from counterfactuals.perturbation_engine import PerturbationSpec

    for bad in [float("nan"), float("inf"), float("-inf")]:
        spec = PerturbationSpec(perturbation_type="cooling", magnitude=bad)
        try:
            spec.resolved_magnitude()
            raise AssertionError(f"magnitude={bad} should have been rejected")
        except ValueError as e:
            msg = str(e).lower()
            assert "finite" in msg, f"error should mention 'finite': {e}"


def test_load_perturbation_defaults_from_config_defaults():
    """Upgrade: load_perturbation_defaults_from_config returns code defaults
    when no config is passed."""
    from counterfactuals.perturbation_engine import (
        load_perturbation_defaults_from_config, _DEFAULT_MAGNITUDES,
    )

    m = load_perturbation_defaults_from_config()
    # YAML-defaults should match code defaults (we just verified this in Probe Q)
    for k, v in _DEFAULT_MAGNITUDES.items():
        assert k in m
        assert abs(m[k] - v) < 1e-9


def test_load_perturbation_defaults_from_config_override():
    """Upgrade: overriding via nested config dict works."""
    from counterfactuals.perturbation_engine import load_perturbation_defaults_from_config

    override = {
        "perturbations": {
            "perturbations": {
                "sleep_extension": {"default": 90.0}
            }
        }
    }
    m = load_perturbation_defaults_from_config(override)
    assert m["sleep_extension"] == 90.0
    # Unknown perturbation type in override is ignored
    override2 = {"perturbations": {"perturbations": {"not_a_real_type": {"default": 999.0}}}}
    m2 = load_perturbation_defaults_from_config(override2)
    assert "not_a_real_type" not in m2


def test_merge_configs_is_non_destructive():
    """Upgrade guard: merge_configs must not mutate its input dicts."""
    from utils.config import merge_configs

    base = {"a": [1, 2, 3], "b": {"x": 1}}
    base_snapshot = {"a": [1, 2, 3], "b": {"x": 1}}
    override = {"b": {"x": 99}}
    merge_configs(base, override)
    assert base["a"] == base_snapshot["a"]
    assert base["b"]["x"] == 1, "merge_configs mutated base"
