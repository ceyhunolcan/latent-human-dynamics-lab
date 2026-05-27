"""Wrapper around `states.latent_state_encoder.MultimodalLatentStateEncoder`.

Centralises the construction of the encoder from a config dict, so the
scripts and the API never have to know the constructor signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from states.latent_state_encoder import MultimodalLatentStateEncoder


@dataclass
class MultimodalEncoderConfig:
    wearable_dim: int = 12
    behavioral_dim: int = 8
    climate_dim: int = 6
    missingness_dim: int = 5
    baseline_dim: int = 8
    proj_dim: int = 32
    hidden_dim: int = 64
    latent_dim: int = 6
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "MultimodalEncoderConfig":
        if not d:
            return cls()
        kwargs = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**kwargs)


def build_multimodal_encoder(config: Optional[MultimodalEncoderConfig] = None):
    """Construct the encoder. Raises ImportError clearly if torch is absent."""
    cfg = config or MultimodalEncoderConfig()
    return MultimodalLatentStateEncoder(
        wearable_dim=cfg.wearable_dim,
        behavioral_dim=cfg.behavioral_dim,
        climate_dim=cfg.climate_dim,
        missingness_dim=cfg.missingness_dim,
        baseline_dim=cfg.baseline_dim,
        proj_dim=cfg.proj_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    )
