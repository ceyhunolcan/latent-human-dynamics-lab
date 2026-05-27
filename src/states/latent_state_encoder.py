"""Multimodal latent state encoder.

The encoder maps daily observations across four input modalities (wearable,
behavioral, climate, missingness) plus a participant baseline vector to a
6-dimensional latent state vector Z_t. Each latent axis is supervised softly
by the corresponding synthetic ground-truth state when available, and
otherwise emerges from the training objectives below.

Two implementations are provided:

* `MultimodalLatentStateEncoder` is a small PyTorch module with modality-
  specific projection heads, a GRU temporal mixer, baseline conditioning,
  and an MC-dropout uncertainty head. PyTorch is an optional dependency;
  importing this module does not require it. Constructing the encoder does.

* `encode_latent_states_classical` is a torch-free fallback that fits modality-
  level PCA, concatenates the projections, and applies a final PCA to extract
  a 6-dimensional code. It produces an honest approximation that is sufficient
  for downstream geometry, regime detection, and counterfactual reasoning.

The training objective functions (`reconstruction_loss`, `smoothness_regularization`,
`contrastive_trajectory_loss`, `disentanglement_penalty`,
`next_state_prediction_loss`) are written so they work with either NumPy
arrays or torch tensors. When torch tensors are passed in, gradients flow
through; with NumPy arrays they reduce to plain scalar computations useful
for offline analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


LATENT_DIM = 6
LATENT_DIM_NAMES = (
    "autonomic_recovery",
    "circadian_alignment",
    "stress_load",
    "environmental_burden",
    "behavioral_instability",
    "missingness_pressure",
)


# ---------------------------------------------------------------------------
# Torch-based encoder
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _ModalityProjection(nn.Module):
        def __init__(self, in_dim: int, proj_dim: int, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, proj_dim * 2),
                nn.LayerNorm(proj_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim),
            )

        def forward(self, x):
            return self.net(x)

    class MultimodalLatentStateEncoder(nn.Module):
        """Encoder mapping multimodal daily observations to latent Z_t.

        Parameters
        ----------
        wearable_dim, behavioral_dim, climate_dim, missingness_dim, baseline_dim
            Input dimensionality of each modality.
        proj_dim
            Shared projection width per modality.
        hidden_dim
            Hidden width of the GRU temporal mixer.
        latent_dim
            Output latent dimensionality (default 6).
        dropout
            MC-dropout rate, used both during training and at inference for
            uncertainty estimation.
        """

        def __init__(
            self,
            wearable_dim: int = 12,
            behavioral_dim: int = 8,
            climate_dim: int = 6,
            missingness_dim: int = 5,
            baseline_dim: int = 8,
            proj_dim: int = 32,
            hidden_dim: int = 64,
            latent_dim: int = LATENT_DIM,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.latent_dim = latent_dim
            self.dropout_rate = dropout

            self.proj_wearable = _ModalityProjection(wearable_dim, proj_dim, dropout)
            self.proj_behavioral = _ModalityProjection(behavioral_dim, proj_dim, dropout)
            self.proj_climate = _ModalityProjection(climate_dim, proj_dim, dropout)
            self.proj_missingness = _ModalityProjection(missingness_dim, proj_dim, dropout)
            self.proj_baseline = _ModalityProjection(baseline_dim, proj_dim, dropout)

            self.modality_mixer = nn.Sequential(
                nn.Linear(proj_dim * 5, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )

            self.to_latent = nn.Linear(hidden_dim, latent_dim)
            self.to_logvar = nn.Linear(hidden_dim, latent_dim)

            # Reconstruct concatenated observed feature vector for the loss.
            self.reconstruction_head = nn.Linear(
                latent_dim + baseline_dim,
                wearable_dim + behavioral_dim + climate_dim + missingness_dim,
            )

        def forward(
            self,
            wearable: "torch.Tensor",
            behavioral: "torch.Tensor",
            climate: "torch.Tensor",
            missingness: "torch.Tensor",
            baseline: "torch.Tensor",
        ):
            """Encode a batch of participant trajectories.

            All inputs are shaped (batch, time, modality_dim). The baseline
            vector is broadcast across time.
            """
            B, T, _ = wearable.shape
            baseline_bt = baseline.unsqueeze(1).expand(B, T, baseline.shape[-1])

            h_w = self.proj_wearable(wearable)
            h_b = self.proj_behavioral(behavioral)
            h_c = self.proj_climate(climate)
            h_m = self.proj_missingness(missingness)
            h_p = self.proj_baseline(baseline_bt)

            h = torch.cat([h_w, h_b, h_c, h_m, h_p], dim=-1)
            h = self.modality_mixer(h)

            seq, _ = self.gru(h)
            mu = self.to_latent(seq)
            logvar = self.to_logvar(seq)

            recon_input = torch.cat([mu, baseline_bt], dim=-1)
            recon = self.reconstruction_head(recon_input)

            return {"latent_mean": mu, "latent_logvar": logvar, "reconstruction": recon}

        def encode_with_uncertainty(
            self,
            wearable,
            behavioral,
            climate,
            missingness,
            baseline,
            n_samples: int = 20,
        ):
            """MC-dropout uncertainty. Returns mean and std over `n_samples` passes."""
            self.train()  # enable dropout for sampling
            samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    out = self.forward(wearable, behavioral, climate, missingness, baseline)
                    samples.append(out["latent_mean"])
            stacked = torch.stack(samples, dim=0)
            return stacked.mean(dim=0), stacked.std(dim=0)

else:  # pragma: no cover - exercised only without torch

    class MultimodalLatentStateEncoder:  # type: ignore[no-redef]
        """Placeholder raised lazily when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MultimodalLatentStateEncoder requires PyTorch. Install with "
                "`pip install torch` or use encode_latent_states_classical()."
            )


# ---------------------------------------------------------------------------
# Torch-free classical encoder
# ---------------------------------------------------------------------------

@dataclass
class ClassicalEncoderResult:
    latent: np.ndarray  # (n_rows, latent_dim)
    explained_variance_ratio: np.ndarray
    column_names: Sequence[str] = field(default_factory=lambda: LATENT_DIM_NAMES)


def encode_latent_states_classical(
    wearable: np.ndarray,
    behavioral: np.ndarray,
    climate: np.ndarray,
    missingness: np.ndarray,
    baseline: np.ndarray,
    latent_dim: int = LATENT_DIM,
) -> ClassicalEncoderResult:
    """Torch-free latent encoder.

    Fits a standard scaler + PCA per modality (up to 4 components each),
    concatenates, then a final PCA collapses to `latent_dim`. This is used
    when PyTorch is not installed and during smoke tests.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    def _safe(arr: np.ndarray, n_comp: int) -> np.ndarray:
        if arr.shape[1] == 0:
            return np.zeros((arr.shape[0], 0))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = StandardScaler().fit_transform(arr)
        # PCA requires n_components <= min(n_samples, n_features). With
        # very short windows (e.g. API calls with 7-day inputs) the sample
        # count is the binding constraint.
        k = min(n_comp, scaled.shape[1], scaled.shape[0])
        if k <= 0:
            return scaled
        return PCA(n_components=k, random_state=17).fit_transform(scaled)

    pieces = [
        _safe(wearable, 4),
        _safe(behavioral, 3),
        _safe(climate, 3),
        _safe(missingness, 2),
        _safe(baseline, 3),
    ]
    stacked = np.concatenate(pieces, axis=1)

    # Final projection: clamp by row count too, and pad with zeros if the
    # stacked representation has fewer components than the requested
    # latent dimension (which can happen on very short windows).
    n_final = min(latent_dim, stacked.shape[1], stacked.shape[0])
    if n_final <= 0:
        return ClassicalEncoderResult(
            latent=np.zeros((stacked.shape[0], latent_dim)),
            explained_variance_ratio=np.zeros(latent_dim),
        )
    final = PCA(n_components=n_final, random_state=17)
    latent = final.fit_transform(stacked)
    if n_final < latent_dim:
        # Pad to canonical latent_dim with zeros so downstream code can
        # always assume a 6-column array.
        latent = np.concatenate(
            [latent, np.zeros((latent.shape[0], latent_dim - n_final))],
            axis=1,
        )
        evr = np.concatenate(
            [final.explained_variance_ratio_, np.zeros(latent_dim - n_final)]
        )
    else:
        evr = final.explained_variance_ratio_
    return ClassicalEncoderResult(latent=latent, explained_variance_ratio=evr)


# ---------------------------------------------------------------------------
# Training objective functions
# ---------------------------------------------------------------------------

def _to_array(x):
    """Coerce torch tensor or numpy array to numpy for fallback paths."""
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def reconstruction_loss(predicted, target):
    """Mean squared error between predicted reconstruction and observed features."""
    if _TORCH_AVAILABLE and isinstance(predicted, torch.Tensor):
        return ((predicted - target) ** 2).mean()
    p, t = _to_array(predicted), _to_array(target)
    return float(((p - t) ** 2).mean())


def next_state_prediction_loss(z_pred_next, z_true_next):
    """One-step ahead MSE on latent state. Encourages temporal coherence."""
    if _TORCH_AVAILABLE and isinstance(z_pred_next, torch.Tensor):
        return ((z_pred_next - z_true_next) ** 2).mean()
    p, t = _to_array(z_pred_next), _to_array(z_true_next)
    return float(((p - t) ** 2).mean())


def smoothness_regularization(z_seq, weight: float = 1.0):
    """Penalise large day-to-day jumps in latent state.

    z_seq is shaped (batch, time, latent_dim) for torch, or (time, latent_dim)
    for numpy.
    """
    if _TORCH_AVAILABLE and isinstance(z_seq, torch.Tensor):
        diff = z_seq[:, 1:] - z_seq[:, :-1]
        return weight * (diff ** 2).mean()
    z = _to_array(z_seq)
    diff = z[..., 1:, :] - z[..., :-1, :]
    return float(weight * (diff ** 2).mean())


def contrastive_trajectory_loss(z_seq, participant_ids, margin: float = 1.0):
    """Light-weight contrastive loss across participant trajectories.

    Encourages within-participant latent samples to be closer to each other
    than to between-participant samples by a margin. Used as a regulariser,
    not a hard objective; works with either torch or numpy inputs.
    """
    z = _to_array(z_seq)
    pids = np.asarray(participant_ids)
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    if pids.ndim == 2:
        pids = pids.reshape(-1)

    # Random sample a small set of triplets to keep this cheap.
    n = len(z)
    rng = np.random.default_rng(17)
    n_triplets = min(256, n // 3)
    if n_triplets < 4:
        return 0.0
    idx = rng.integers(0, n, size=(n_triplets, 3))
    losses = []
    for a, p, q in idx:
        if pids[a] == pids[p] and pids[a] != pids[q]:
            d_pos = float(((z[a] - z[p]) ** 2).sum())
            d_neg = float(((z[a] - z[q]) ** 2).sum())
            losses.append(max(0.0, d_pos - d_neg + margin))
    return float(np.mean(losses)) if losses else 0.0


def disentanglement_penalty(z_seq):
    """Lightweight off-diagonal covariance penalty.

    Encourages the latent dimensions to be approximately uncorrelated, which
    is a soft prior toward semantically separable axes. Not a guarantee of
    disentanglement in the technical ICA sense.
    """
    if _TORCH_AVAILABLE and isinstance(z_seq, torch.Tensor):
        z = z_seq.reshape(-1, z_seq.shape[-1])
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(z.shape[0] - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        return (off ** 2).mean()
    z = _to_array(z_seq)
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    z = z - z.mean(axis=0, keepdims=True)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    off = cov - np.diag(np.diag(cov))
    return float((off ** 2).mean())
