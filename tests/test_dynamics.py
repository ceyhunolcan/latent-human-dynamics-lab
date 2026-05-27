"""Tests for the dynamics model and perturbation operator."""

import numpy as np


def test_dynamics_predict_next_state_shape():
    from dynamics.transition_model import LatentDynamicsModel

    model = LatentDynamicsModel(rng_seed=17)
    z = np.zeros(6)
    b = np.zeros(4)
    e = np.zeros(4)
    z_next = model.predict_next_state(z, b, e, perturbation_t=None)
    assert z_next.shape == (6,)
    assert np.isfinite(z_next).all()


def test_dynamics_simulate_trajectory_length():
    from dynamics.transition_model import LatentDynamicsModel

    model = LatentDynamicsModel(rng_seed=17)
    z0 = np.zeros(6)
    env = np.zeros((10, 4))
    beh = np.zeros((10, 4))
    traj = model.simulate_trajectory(z0, environment=env, behavior=beh)
    assert traj.shape == (11, 6)  # T+1


def test_apply_perturbation_operator_signs():
    """Cooling should reduce environmental burden; heat wave should increase it."""
    from dynamics.forcing_functions import apply_perturbation_operator

    cool = apply_perturbation_operator(
        Z=np.zeros((14, 6)),
        perturbation_type="cooling",
        magnitude=-4.0,
        horizon_days=14,
    )
    heat = apply_perturbation_operator(
        Z=np.zeros((14, 6)),
        perturbation_type="heat_wave_shock",
        magnitude=6.0,
        horizon_days=14,
    )
    # Index 3 = environmental_burden
    assert cool[0, 3] < 0, "Cooling should reduce environmental burden"
    assert heat[-1, 3] > 0, "Heat wave should increase environmental burden over time"


def test_environmental_forcing_shape():
    from dynamics.forcing_functions import environmental_forcing

    daytime = np.array([28.0, 30.0])
    nighttime = np.array([22.0, 24.0])
    aqi = np.array([80.0, 100.0])
    heatwave = np.array([0.0, 1.0])
    forcing = environmental_forcing(daytime, nighttime, aqi, heatwave)
    assert forcing.shape == (2, 6)


def test_resilience_profile_dataclass():
    import pandas as pd

    from dynamics.resilience_model import estimate_resilience_profile

    df = pd.DataFrame(
        {
            "participant_id": ["P001"] * 30,
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "stress_score": np.random.RandomState(0).rand(30),
            "recovery_score": 70 + np.random.RandomState(1).randn(30) * 5,
        }
    )
    profile = estimate_resilience_profile(df, participant_id="P001")
    assert profile.participant_id == "P001"
    assert profile.profile_label in (
        "high_resilience",
        "moderate_resilience",
        "climate_vulnerable",
        "recovery_delayed",
        "behaviorally_unstable",
    )


def test_combined_resilience_protocol_respects_magnitude():
    """Regression test for Bug 6: combined branch ignored magnitude argument."""
    from dynamics.forcing_functions import apply_perturbation_operator

    d1 = apply_perturbation_operator(
        np.zeros((14, 6)), "combined_resilience_protocol", 1.0, 14
    )
    d10 = apply_perturbation_operator(
        np.zeros((14, 6)), "combined_resilience_protocol", 10.0, 14
    )
    ratio = np.abs(d10).max() / max(np.abs(d1).max(), 1e-10)
    assert 9.5 < ratio < 10.5, f"magnitude scaling broken: ratio={ratio:.2f}"


def test_dynamics_bounded_on_raw_cohort_inputs():
    """Regression test for Bug 7: dynamics exploded on raw (non-z-scored) inputs.

    The forcing functions used to assume z-scored inputs but every caller in
    the repo was passing raw values like temperature_c=28.0 and
    screen_time_minutes=240.0. Trajectories then grew unbounded over a
    14-day horizon. After the fix the dynamics layer normalizes internally.
    """
    from dynamics.transition_model import LatentDynamicsModel

    m = LatentDynamicsModel(rng_seed=17)
    z0 = np.zeros(6)
    # Raw cohort-scale values, NOT z-scored
    env = np.tile([[28.0, 22.0, 80.0, 0.0]], (14, 1))
    beh = np.tile([[240.0, 5.0, 0.6, 80.0]], (14, 1))
    traj = m.simulate_trajectory(z0, env, beh)
    assert traj.shape == (15, 6)
    assert np.abs(traj).max() < 5.0, (
        f"trajectory should stay bounded but max |z| = {np.abs(traj).max():.2f}"
    )


def test_environmental_forcing_handles_raw_inputs():
    """Regression test: environmental_forcing(normalize_inputs=True) produces
    bounded output on raw cohort-scale temperatures and AQI."""
    from dynamics.forcing_functions import environmental_forcing

    F = environmental_forcing(
        daytime_heat=np.array([28.0, 35.0]),     # raw °C
        nighttime_heat=np.array([22.0, 28.0]),   # raw °C
        aqi=np.array([80.0, 150.0]),             # raw AQI
        heatwave_exposure_days=np.array([0.0, 3.0]),
        normalize_inputs=True,
    )
    assert F.shape == (2, 6)
    assert np.abs(F).max() < 5.0, f"forcing too large: max={np.abs(F).max():.2f}"


def test_recovery_half_life_recovers_clean_exponential():
    """Regression for Bug 13: recovery_half_life was shifting the input
    by its minimum before taking log, which destroyed the exponential
    structure. On a clean exponential with half-life 3, the estimator
    used to return ≈4.65; it should return ≈3.0.
    """
    from dynamics.resilience_model import recovery_half_life

    for true_tau in [2.0, 3.0, 5.0, 7.0]:
        t = np.arange(50)
        stress = np.exp(-t * np.log(2) / true_tau)
        est = recovery_half_life(stress)
        assert abs(est - true_tau) < 0.1, (
            f"true_τ={true_tau}, estimated {est:.3f}"
        )


def test_recovery_half_life_handles_pathological_inputs():
    """recovery_half_life should not crash and should clip to plausible
    range on degenerate or adversarial inputs."""
    from dynamics.resilience_model import recovery_half_life

    # Constant — no decay information
    assert 0.5 <= recovery_half_life(np.ones(50)) <= 21.0
    # Increasing — opposite of recovery
    assert 0.5 <= recovery_half_life(np.arange(50, dtype=float) + 1) <= 21.0
    # Negative-valued signal (after shift)
    stress = np.exp(-np.arange(50) * np.log(2) / 3.0) - 0.5
    assert 0.5 <= recovery_half_life(stress) <= 21.0


def test_high_contraction_decays_under_neutral_inputs():
    """With high contraction and inputs at the normalization reference points,
    trajectory should decay to ~0 quickly."""
    from dynamics.transition_model import LatentDynamicsModel
    import numpy as np

    m = LatentDynamicsModel(contraction=0.9, noise_std=0.0, rng_seed=17)
    z0 = np.array([5.0] * 6)
    # Neutral env values matching the normalization reference points
    env = np.tile([18.0, 14.0, 50.0, 0.0], (30, 1))
    beh = np.zeros((30, 4))
    traj = m.simulate_trajectory(z0, env, beh)
    assert np.abs(traj[-1]).max() < 0.01, (
        f"contraction=0.9 should drive state to 0; got |z| = {np.abs(traj[-1]).max():.4f}"
    )


def test_zero_contraction_preserves_state_under_neutral_inputs():
    """With contraction=0 and zero forcing, state should be preserved exactly."""
    from dynamics.transition_model import LatentDynamicsModel
    import numpy as np

    m = LatentDynamicsModel(contraction=0.0, noise_std=0.0, rng_seed=17)
    z0 = np.array([1.0] * 6)
    # Neutral env values
    env = np.tile([18.0, 14.0, 50.0, 0.0], (30, 1))
    beh = np.zeros((30, 4))
    traj = m.simulate_trajectory(z0, env, beh)
    assert np.abs(traj[-1] - z0).max() < 0.01, (
        f"zero contraction with neutral inputs should preserve state; final={traj[-1]}"
    )


def test_load_epl_weights_from_config_defaults():
    """Upgrade: load_epl_weights_from_config returns defaults when no config
    is passed, and falls back to defaults on garbage input."""
    from dynamics.forcing_functions import load_epl_weights_from_config, EPL_DEFAULT_WEIGHTS

    # No config → defaults
    w = load_epl_weights_from_config()
    assert w == EPL_DEFAULT_WEIGHTS or all(w[k] == EPL_DEFAULT_WEIGHTS[k] for k in EPL_DEFAULT_WEIGHTS)

    # Garbage config → defaults
    w = load_epl_weights_from_config({"unrelated": True})
    assert w == EPL_DEFAULT_WEIGHTS


def test_load_epl_weights_from_config_override():
    """Upgrade: overriding via nested config dict works."""
    from dynamics.forcing_functions import load_epl_weights_from_config

    override = {"dynamics": {"forcing": {"epl_weights": {"aqi": 0.5}}}}
    w = load_epl_weights_from_config(override)
    assert w["aqi"] == 0.5
    # Non-overridden keys keep defaults
    assert w["daytime_heat"] == 0.30
    assert w["nighttime_heat"] == 0.30
    assert w["heatwave"] == 0.15


def test_vulnerability_nan_rejected():
    """Regression for Bug 20: NaN vulnerability_coefficient used to
    silently propagate to the latent state, producing all-NaN output."""
    import numpy as np
    from dynamics.transition_model import LatentDynamicsModel

    m = LatentDynamicsModel()
    for bad in [float("nan"), float("inf"), float("-inf")]:
        try:
            m.predict_next_state(
                np.zeros(6), np.zeros(4), np.array([20., 14., 50., 0.]),
                vulnerability_coefficient=bad,
            )
            raise AssertionError(f"vulnerability={bad} should have been rejected")
        except ValueError as e:
            assert "finite" in str(e).lower()


def test_dynamics_from_config_picks_up_yaml_defaults():
    """Upgrade verification: LatentDynamicsModel.from_config() should
    read configs/dynamics.yaml and not just use code defaults."""
    from dynamics.transition_model import LatentDynamicsModel
    from utils.config import get_dynamics_settings

    # Default config produces the paper-canonical values
    m = LatentDynamicsModel.from_config()
    settings = get_dynamics_settings()
    assert abs(m.contraction - settings["contraction"]) < 1e-9
    assert abs(m.noise_std - settings["noise_std"]) < 1e-9


def test_dynamics_from_config_respects_overrides():
    """Custom config dict should override defaults."""
    from dynamics.transition_model import LatentDynamicsModel

    custom = {
        "dynamics": {
            "transition": {"contraction": 0.25},
            "resilience": {"noise_sigma": 0.07},
        }
    }
    m = LatentDynamicsModel.from_config(custom)
    assert abs(m.contraction - 0.25) < 1e-9
    assert abs(m.noise_std - 0.07) < 1e-9


def test_nan_env_inputs_handled_gracefully():
    """Regression for Bug 22: NaN in env/behavior used to silently propagate
    to the latent state. They should now be replaced with the
    normalization-reference values so the post-z-score signal is zero
    (== "no information available, neutral day").
    """
    import numpy as np
    from dynamics.transition_model import LatentDynamicsModel

    m = LatentDynamicsModel(rng_seed=17, noise_std=0.0)
    z0 = np.zeros(6)

    # All NaN env should not propagate
    z_next = m.predict_next_state(z0, np.zeros(4), np.array([np.nan]*4))
    assert np.isfinite(z_next).all(), f"NaN env propagated: {z_next}"

    # Mixed NaN/finite env
    z_next = m.predict_next_state(z0, np.zeros(4), np.array([np.nan, 14.0, np.nan, 0.0]))
    assert np.isfinite(z_next).all(), f"mixed NaN env propagated: {z_next}"

    # NaN behavior should also not propagate
    z_next = m.predict_next_state(z0, np.array([np.nan, 0, 0, 0]), np.array([18.0, 14.0, 50.0, 0.0]))
    assert np.isfinite(z_next).all(), f"NaN behavior propagated: {z_next}"


def test_nan_latent_state_rejected():
    """Z_t (latent state input) MUST be finite. Unlike env/behavior, there's
    no sensible neutral imputation — NaN in Z means the upstream encoder
    already failed. The dynamics layer rejects this loudly."""
    import numpy as np
    from dynamics.transition_model import LatentDynamicsModel

    m = LatentDynamicsModel(rng_seed=17)
    try:
        m.predict_next_state(np.array([np.nan]*6), np.zeros(4), np.zeros(4))
        raise AssertionError("NaN Z_t should have been rejected")
    except ValueError as e:
        msg = str(e).lower()
        assert "finite" in msg or "z_t" in msg
