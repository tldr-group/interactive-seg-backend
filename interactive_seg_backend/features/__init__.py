__all__ = [
    "multiscale_features",
    "singlescale_singlechannel_features",
    "TORCH_AVAILABLE",
    "multiscale_features_gpu",
    "prepare_for_gpu",
    "concat_feats",
    "N_ALLOWED_CPUS",
]
from .multiscale_classical_cpu import multiscale_features, singlescale_singlechannel_features, N_ALLOWED_CPUS
from .multiscale_classical_gpu import (
    TORCH_AVAILABLE,
    multiscale_features_gpu,
    prepare_for_gpu,
    concat_feats,
)
