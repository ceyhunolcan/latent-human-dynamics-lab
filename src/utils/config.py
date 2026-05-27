"""YAML configuration loading + simple deep-merge."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .paths import CONFIGS_DIR


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain dict. Empty / missing yields {}."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_default_config() -> Dict[str, Any]:
    """Load all default configs and stitch them under top-level keys."""
    cfg: Dict[str, Any] = {}
    for name in ("default", "model", "dynamics", "perturbations", "experiments"):
        path = CONFIGS_DIR / f"{name}.yaml"
        cfg[name] = load_yaml(path)
    return cfg


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts. Values in ``override`` win on conflict."""
    result = deepcopy(base)
    for k, v in (override or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def get_dynamics_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Pull dynamics model settings from the config stack.

    Returns a dict with keys ``contraction``, ``noise_std``, ``forcing_scale``,
    and ``base_half_life_days``. If the config file is missing or the keys
    aren't present, returns the canonical defaults that match the paper.
    """
    if cfg is None:
        cfg = load_default_config()
    dyn = cfg.get("dynamics", {}) or {}
    resilience = dyn.get("resilience", {}) or {}
    transition = dyn.get("transition", {}) or {}
    return {
        "contraction": float(transition.get("contraction", 0.1)),
        "noise_std": float(resilience.get("noise_sigma", 0.01)),
        "forcing_scale": float(transition.get("forcing_scale", 0.04)),
        "base_half_life_days": float(resilience.get("base_half_life_days", 4.0)),
    }


def get_epl_weights(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Pull EPL weights from the config stack.

    Falls back to the canonical (0.30, 0.30, 0.25, 0.15) when the config is
    missing or partial.
    """
    if cfg is None:
        cfg = load_default_config()
    dyn = cfg.get("dynamics", {}) or {}
    weights = (dyn.get("forcing", {}) or {}).get("epl_weights", {})
    canonical = {
        "daytime_heat": 0.30,
        "nighttime_heat": 0.30,
        "aqi": 0.25,
        "heatwave": 0.15,
    }
    return {k: float(weights.get(k, v)) for k, v in canonical.items()}


def get_perturbation_defaults(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Pull per-perturbation default magnitudes from configs/perturbations.yaml.

    Falls back to the code-default if YAML doesn't override.
    """
    if cfg is None:
        cfg = load_default_config()
    pert = (cfg.get("perturbations", {}) or {}).get("perturbations", {})
    canonical = {
        "sleep_extension": 45.0,
        "screen_reduction": 60.0,
        "exercise_boost": 30.0,
        "cooling": -4.0,
        "air_quality_improvement": -40.0,
        "heat_wave_shock": 6.0,
        "combined_resilience_protocol": 1.0,
    }
    return {
        k: float((pert.get(k) or {}).get("default", v))
        for k, v in canonical.items()
    }
