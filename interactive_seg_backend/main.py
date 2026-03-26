import numpy as np
from typing import Callable, TypeAlias

from interactive_seg_backend.configs.config import FeatureConfig, TrainingConfig
from interactive_seg_backend.extensions.autocontext import autocontext_features
from interactive_seg_backend.file_handling import save_featurestack
from interactive_seg_backend.features import (
    prepare_for_gpu,
    transfer_from_gpu,
    concat_feats,
)
from interactive_seg_backend.configs.types import AnyArr, Arrlike, NPFloatArray, NPUIntArray

from interactive_seg_backend.classifiers import Classifier
from interactive_seg_backend.core import (
    get_labelled_training_data_from_stack,
    shuffle_sample_training_data,
    get_model,
    train,
    apply_,
    featurise_,
    train_and_apply_,
)
from interactive_seg_backend.extensions.crf import (
    do_crf_from_probabilites,
    CRF_AVAILABLE,
)
from interactive_seg_backend.processing.postprocess import modal_filter


FeatureFunction: TypeAlias = Callable[[Arrlike, FeatureConfig], Arrlike]


def featurise(
    image: NPFloatArray,
    training_cfg: TrainingConfig,
    save_path: str = "",
    custom_fns: list[tuple[FeatureFunction, bool]] = [],
) -> AnyArr:
    """Compute features of $image according to feature_config of $training_cfg. Optionally store at $save path.
    You can also supply custom feature functions in $custom_fns, which should be a list of tuples of the form
    (fn, gpu_flag) where fn is a function that takes an image and a feature config and returns an array of features,
    and gpu_flag is a boolean that indicates whether the function should be run on the GPU or not.

    Args:
        image (NPFloatArray): (H,W) or (H,W,C) array of image to featureise. Can be uint or float.
        training_cfg (TrainingConfig): contains feature_config that specifies features to compute & with what parameters.
        save_path (str, optional): optional path to cache features to. Defaults to "".
        custom_fns (list[tuple[FeatureFunction, bool]], optional): optional additional featurisation functions. Defaults to [].

    Returns:
        AnyArr: (H,W,C) array of features, where C is the total number of features computed by both featurise_ and custom_fns.
    """
    feature_cfg = training_cfg.feature_config

    feats = featurise_(
        image,
        training_cfg.feature_config,
        training_cfg.preprocessing,
        training_cfg.use_gpu,
    )

    for fn, gpu_flag in custom_fns:
        if gpu_flag:
            tensor = prepare_for_gpu(image)
            custom_feats = fn(tensor, feature_cfg)
        else:
            custom_feats = fn(image, feature_cfg)
        feats = concat_feats(feats, custom_feats)

    if save_path != "":
        save_featurestack(feats, save_path, ".npy")
    return feats


def apply(
    model: Classifier,
    features: NPFloatArray,
    training_cfg: TrainingConfig,
    image: np.ndarray | None = None,
    labels: NPUIntArray | None = None,
) -> tuple[NPUIntArray, NPFloatArray]:
    """Apply trained $model to $features, applying additional post-processing as defined in $training_cfg.
    This generates a (H,W,N_classes) probability map which is then argmaxed to get final segmentation.
    Optionally applies modal filter, CRF and/or autocontext post-processing if specified in $training_cfg.

    Args:
        model (Classifier): trained classifier.
        features (NPFloatArray): (H,W,C) array of features to apply model to.
            NB: must be in same order and distribution as features used to train model.
        training_cfg (TrainingConfig): config that describes post-processing to apply to raw model probabilities, e.g. CRF or autocontext.
        image (np.ndarray | None, optional): image $features are from. Required for CRF. Defaults to None.
        labels (NPUIntArray | None, optional): labels $classifier was trained with. Required for autocontext. Defaults to None.

    Returns:
        tuple[NPUIntArray, NPFloatArray]: (H,W) label array of segmentation and (H,W,N_classes) array of class probabilities.
    """
    seg, probs_2D = apply_(model, features)
    _, _, n_classes = probs_2D.shape

    if training_cfg.autocontext and CRF_AVAILABLE:
        assert image is not None, "Need Image to do autocontext"
        assert labels is not None, "Need labels to do autocontext"
        new_feats = autocontext_features(image, labels, training_cfg, features, probs_2D, "autocontext_original")
        new_feats = transfer_from_gpu(new_feats)
        seg, probs_2D, _ = train_and_apply_(new_feats, labels, training_cfg)

    if training_cfg.CRF and CRF_AVAILABLE:
        assert image is not None, "Need Image to do CRF"
        params = training_cfg.CRF_params
        seg = do_crf_from_probabilites(probs_2D, image, n_classes, params)

    if training_cfg.modal_filter:
        seg = modal_filter(seg, training_cfg.modal_filter_k)

    return seg, probs_2D


def train_and_apply(
    features: NPFloatArray,
    labels: NPUIntArray,
    train_cfg: TrainingConfig,
    image: np.ndarray | None = None,
) -> tuple[NPUIntArray, NPFloatArray, Classifier]:
    """End-to-end train and apply model based on $features and $labels.

    Args:
        features (NPFloatArray): (H,W,C) array of features to apply model to.
        training_cfg (TrainingConfig): config that describes post-processing to apply to raw model probabilities, e.g. CRF or autocontext.
        image (np.ndarray | None, optional): image $features are from. Required for CRF. Defaults to None.

    Returns:
        tuple[NPUIntArray, NPFloatArray]: (H,W) label array of segmentation and (H,W,N_classes) array of class probabilities.
    """
    fit, target = get_labelled_training_data_from_stack(features, labels)
    fit, target = shuffle_sample_training_data(fit, target, train_cfg.shuffle_data, train_cfg.n_samples)
    model = get_model(train_cfg.classifier, train_cfg.classifier_params, train_cfg.use_gpu)
    model = train(model, fit, target, None)
    pred, probs = apply(model, features, train_cfg, labels=labels, image=image)
    return pred, probs, model
