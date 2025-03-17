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
    which: AutocontextType = "autocontext_ilastik",
) -> AnyArr:
    original_feats = featurise_(
        image, train_cfg.feature_config, train_cfg.preprocessing, train_cfg.use_gpu
    )
    _, probs, _ = train_and_apply_(original_feats, labels, train_cfg)
    if which == "autocontext_original":
        return concat_feats(original_feats, probs)
    else:
        prob_feats = featurise_(
            probs, train_cfg.feature_config, train_cfg.preprocessing, train_cfg.use_gpu
        )
        return concat_feats(original_feats, prob_feats)
