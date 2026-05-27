"""Temporal transformer.

A drop-in alternative to the GRU temporal mixer in the multimodal encoder.
Useful when sequences are long enough that attention beats recurrence, or
when interpretability via attention maps is desired.
"""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 1024):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)].to(x.device)


    class TemporalTransformer(nn.Module):
        """Small encoder-only transformer for daily multimodal sequences."""

        def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            d_ff: int = 128,
            dropout: float = 0.1,
            max_len: int = 365,
        ):
            super().__init__()
            self.pos = _PositionalEncoding(d_model, max_len=max_len)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        def forward(self, x, src_key_padding_mask=None):
            x = self.pos(x)
            return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

else:  # pragma: no cover

    class TemporalTransformer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("TemporalTransformer requires PyTorch.")


def build_temporal_transformer(d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
    """Convenience constructor matching the config style of other modules."""
    return TemporalTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
