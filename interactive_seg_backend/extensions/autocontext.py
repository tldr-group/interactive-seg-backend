from interactive_seg_backend.core import featurise_, train_and_apply_
from interactive_seg_backend.features import concat_feats
from interactive_seg_backend.configs import (
    Arr,
    AnyArr,
    UInt8Arr,
    TrainingConfig,
)

from typing import Literal

AutocontextType = Literal["autocontext_original", "autocontext_ilastik"]


def autocontext_features(
    image: Arr,
    labels: UInt8Arr,
    train_cfg: TrainingConfig,
    original_feats: Arr | None = None,
    original_probs: Arr | None = None,
    which: AutocontextType = "autocontext_ilastik",
) -> AnyArr:
    if original_feats is None:
        feats = featurise_(
            image, train_cfg.feature_config, train_cfg.preprocessing, train_cfg.use_gpu
        )
    else:
        feats = original_feats

    if original_probs is None:
        _, probs, _ = train_and_apply_(feats, labels, train_cfg)
    else:
        probs = original_probs

    if which == "autocontext_original":
        return concat_feats(feats, probs)
    else:
        prob_feats = featurise_(
            probs, train_cfg.feature_config, train_cfg.preprocessing, train_cfg.use_gpu
        )
        return concat_feats(feats, prob_feats)
