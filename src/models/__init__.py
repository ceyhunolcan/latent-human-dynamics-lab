"""Supporting model layer.

Contains the larger torch modules and prediction heads used by the training
scripts and the API: the multimodal encoder wrapper, a temporal transformer
alternative to the GRU mixer, uncertainty quantification heads, and a
set of downstream task models that operate on encoded latent states.

PyTorch is an optional dependency. All modules degrade gracefully when it
is absent; the classes either raise on construction or expose numpy-only
fallbacks.
"""

from .multimodal_encoder import (
    MultimodalEncoderConfig,
    build_multimodal_encoder,
)
from .temporal_transformer import (
    TemporalTransformer,
    build_temporal_transformer,
)
from .uncertainty_heads import (
    MCDropoutHead,
    EnsembleLiteHead,
    prediction_interval,
    classify_uncertainty,
)
from .downstream_tasks import (
    DownstreamTaskHead,
    train_logistic_baseline,
    evaluate_baseline,
)

__all__ = [
    "MultimodalEncoderConfig",
    "build_multimodal_encoder",
    "TemporalTransformer",
    "build_temporal_transformer",
    "MCDropoutHead",
    "EnsembleLiteHead",
    "prediction_interval",
    "classify_uncertainty",
    "DownstreamTaskHead",
    "train_logistic_baseline",
    "evaluate_baseline",
]
