"""Uncertainty quantification.

Two complementary approaches, both used elsewhere in the package:

* `MCDropoutHead` keeps dropout active at inference time and aggregates over
  forward passes. It is the cheapest source of epistemic uncertainty in this
  codebase.

* `EnsembleLiteHead` averages a small bag of independently-initialised
  predictors. Slower than MC dropout but more reliable when the predictions
  span an ill-conditioned region of input space.

Both produce a per-prediction mean and standard deviation. `prediction_interval`
turns those into a symmetric interval at a given confidence level, and
`classify_uncertainty` returns a qualitative flag ("low" / "moderate" /
"high") that the API and dashboard surface to users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class MCDropoutHead(nn.Module):
        """MLP head that keeps dropout active during inference."""

        def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
            self.dropout = dropout

        def forward(self, x):
            return self.net(x)

        def predict_with_uncertainty(self, x, n_samples: int = 20) -> Tuple["torch.Tensor", "torch.Tensor"]:
            self.train()  # keep dropout
            samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    samples.append(self.forward(x))
            stacked = torch.stack(samples, dim=0)
            return stacked.mean(dim=0), stacked.std(dim=0)

else:  # pragma: no cover

    class MCDropoutHead:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("MCDropoutHead requires PyTorch.")


@dataclass
class EnsembleLiteHead:
    """Ensemble of callables. Each callable maps x -> array.

    The class is torch-free so it can be used to wrap arbitrary numpy or
    scikit-learn predictors, e.g. a bag of bootstrapped logistic regressions.
    """

    predictors: List[Callable]

    def predict_with_uncertainty(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.predictors:
            raise ValueError("ensemble is empty")
        samples = np.stack([np.asarray(p(x)) for p in self.predictors], axis=0)
        return samples.mean(axis=0), samples.std(axis=0)


def prediction_interval(
    mean: np.ndarray, std: np.ndarray, confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric prediction interval assuming Gaussian noise."""
    # 95% -> 1.96; 90% -> 1.645; 99% -> 2.576
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    return mean - z * std, mean + z * std


def classify_uncertainty(std: np.ndarray, low: float = 0.05, high: float = 0.20) -> np.ndarray:
    """Return a per-prediction qualitative flag.

    Defaults assume the prediction range is roughly unit-scaled. For unscaled
    predictions the caller should pass appropriate thresholds.
    """
    s = np.atleast_1d(np.asarray(std, dtype=float))
    flags = np.full(s.shape, "moderate", dtype=object)
    flags[s < low] = "low"
    flags[s >= high] = "high"
    return flags
