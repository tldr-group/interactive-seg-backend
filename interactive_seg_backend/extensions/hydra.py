import numpy as np
from interactive_seg_backend.configs import TrainingConfig, CRFParams
from interactive_seg_backend.configs.types import Arr, UInt8Arr
from interactive_seg_backend.features import (
    concat_feats,
)
from interactive_seg_backend.extensions.crf import do_crf_from_probabilites
from interactive_seg_backend.processing.postprocess import modal_filter
from interactive_seg_backend.core import (
    get_labelled_training_data_from_stack,
    get_model,
    shuffle_sample_training_data,
    train,
    apply_,
)


# make sure classifier sees all feats (linear/full XGB/LR)
# combine naive seg (onehot), probs2D, crf_seg, crf_probs, modal filter (onehot)


def apply_hydra(
    original_seg: UInt8Arr,
    original_probs: Arr,
    train_cfg: TrainingConfig,
    image: np.ndarray,
    labels: UInt8Arr,
    crf_seg: Arr | None = None,
    crf_probs: Arr | None = None,
    modal_seg: UInt8Arr | None = None,
) -> tuple[UInt8Arr, Arr]:
    _, _, n_classes = original_probs.shape

    # if crf_seg is None or crf_probs is None:
    fine_crf_params = CRFParams(sxy_g=(1, 1))
    fine_crf_seg, fine_crf_probs = do_crf_from_probabilites(
        original_probs, image, n_classes, fine_crf_params
    )

    coarse_crf_params = CRFParams(sxy_g=(3, 3))
    coarse_crf_seg, coarse_crf_probs = do_crf_from_probabilites(
        original_probs, image, n_classes, coarse_crf_params
    )

    if modal_seg is None:
        modal_seg = modal_filter(original_seg, train_cfg.modal_filter_k)

    feats = concat_feats(np.expand_dims(original_seg, -1), original_probs)
    feats = concat_feats(feats, fine_crf_probs)
    feats = concat_feats(feats, np.expand_dims(fine_crf_seg, -1))
    feats = concat_feats(feats, coarse_crf_probs)
    feats = concat_feats(feats, np.expand_dims(coarse_crf_seg, -1))
    feats = concat_feats(feats, np.expand_dims(modal_seg, -1))

    fit, target = get_labelled_training_data_from_stack(feats, labels)
    fit, target = shuffle_sample_training_data(
        fit, target, train_cfg.shuffle_data, train_cfg.n_samples
    )
    # model = get_model("linear_regression", {}, False)
    model = get_model("random_forest", {"max_features": None, "max_depth": 10}, False)
    model = train(model, fit, target, None)

    seg, probs = apply_(model, feats)

    return seg, probs
