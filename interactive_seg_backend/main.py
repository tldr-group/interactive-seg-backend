import numpy as np
from typing import Callable, TypeAlias
from PIL import Image

from interactive_seg_backend.configs.config import FeatureConfig, TrainingConfig
from interactive_seg_backend.extensions.autocontext import autocontext_features
from interactive_seg_backend.file_handling import save_featurestack
from interactive_seg_backend.features import (
    prepare_for_gpu,
    concat_feats,
)
from interactive_seg_backend.configs.types import (
    Arr,
    AnyArr,
    UInt8Arr,
    Arrlike,
    # UInt8Arrlike,
    # ClassifierNames,
)

from interactive_seg_backend.classifiers import (
    Classifier,
    # RandomForest,
    # Logistic,
    # Linear,
    # XGBCPU,
    # XGBGPU,
)
from interactive_seg_backend.core import (
    get_labelled_training_data_from_stack,
    shuffle_sample_training_data,
    get_model,
    train,
    apply_,
    featurise_,
    train_and_apply_,
)
from interactive_seg_backend.extensions.crf import do_crf_from_probabilites, CRFParams
from interactive_seg_backend.processing.postprocess import modal_filter
from interactive_seg_backend.extensions.hydra import apply_hydra

FeatureFunction: TypeAlias = Callable[[Arrlike, FeatureConfig], Arrlike]


def featurise(
    image: Arr,
    training_cfg: TrainingConfig,
    use_gpu: bool = False,
    save_path: str = "",
    custom_fns: list[tuple[FeatureFunction, bool]] = [],
) -> AnyArr:
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
    features: Arrlike,
    training_cfg: TrainingConfig,
    h: int | None = None,
    w: int | None = None,
    image: np.ndarray | None = None,
    labels: UInt8Arr | None = None,
) -> tuple[UInt8Arr, Arr]:
    seg, probs_2D = apply_(model, features)
    _, _, n_classes = probs_2D.shape

    if training_cfg.autocontext:
        assert labels is not None, "Need labels to do CRF"
        new_feats = autocontext_features(image, labels, training_cfg, features, probs_2D, "autocontext_original")
        seg, probs_2D, _ = train_and_apply_(new_feats, labels, training_cfg)

    crf_probs = None
    img_arr = np.array(Image.fromarray(image).convert("RGB"))
    if training_cfg.CRF:
        assert image is not None, "Need Image to do CRF"
        params = training_cfg.CRF_params
        seg, crf_probs = do_crf_from_probabilites(probs_2D, img_arr, n_classes, params)

    if training_cfg.CRF_AC:
        # assert crf_probs is not None, "Need CRF to do CRF_AC"
        seg, probs_2D = apply_hydra(seg, probs_2D, training_cfg, img_arr, labels)

    if training_cfg.modal_filter:
        seg = modal_filter(seg, training_cfg.modal_filter_k)

    return seg, probs_2D


def train_and_apply(features: Arrlike, labels: UInt8Arr, train_cfg: TrainingConfig) -> tuple[UInt8Arr, Arr, Classifier]:
    fit, target = get_labelled_training_data_from_stack(features, labels)
    fit, target = shuffle_sample_training_data(fit, target, train_cfg.shuffle_data, train_cfg.n_samples)
    model = get_model(train_cfg.classifier, train_cfg.classifier_params, train_cfg.use_gpu)
    model = train(model, fit, target, None)
    pred, probs = apply(model, features, train_cfg, labels=labels)
    return pred, probs, model
