import numpy as np
from interactive_seg_backend.configs.config import FeatureConfig, TrainingConfig
from interactive_seg_backend.features import (
    multiscale_features,
    multiscale_features_gpu,
    prepare_for_gpu,
)
from interactive_seg_backend.configs.types import (
    Arr,
    UInt8Arr,
    Arrlike,
    UInt8Arrlike,
    ClassifierNames,
)
from interactive_seg_backend.classifiers import (
    Classifier,
    RandomForest,
    Logistic,
    Linear,
    XGBCPU,
    XGBGPU,
)

from typing import Any


def get_labelled_training_data_from_stack(
    feature_stack: Arrlike, labels: UInt8Arr
) -> tuple[Arrlike, UInt8Arr]:
    h, w, n_feats = feature_stack.shape
    flat_labels = labels.reshape((h * w))
    flat_features = feature_stack.reshape((h * w, n_feats))

    labelled_mask = np.nonzero(flat_labels)

    fit_data = flat_features[labelled_mask[0], :]
    target_data = flat_labels[labelled_mask[0]]

    return fit_data, target_data


def shuffle_sample_training_data(
    fit: Arrlike, target: UInt8Arrlike, shuffle: bool = True, sample_n: int = -1
) -> tuple[Arrlike, UInt8Arrlike]:
    n_samples = target.shape[0]
    all_inds = np.arange(0, n_samples, 1)
    if shuffle:
        np.random.shuffle(all_inds)
    if sample_n > 0:
        sample_inds = all_inds[:sample_n]
        return fit[sample_inds], target[sample_inds]
    else:
        return fit[all_inds], target[all_inds]


def get_model(
    model_type: ClassifierNames, extra_args: dict[str, Any], to_gpu: bool = False
) -> Classifier:
    if model_type == "random_forest":
        return RandomForest(extra_args)
    elif model_type == "logistic_regression":
        return Logistic(extra_args)
    elif model_type == "linear_regression":
        return Linear(extra_args)
    elif model_type == "xgb" and to_gpu is False:
        return XGBCPU(extra_args)
    elif model_type == "xgb" and to_gpu is True:
        return XGBGPU(extra_args)
    else:
        raise Exception("Not implemented!")


def train(
    model: Classifier, fit: Arrlike, target: UInt8Arr, sample_weight: Arr | None
) -> Classifier:
    if sample_weight is None:
        model.fit(fit, target)
        return model
    else:
        model.fit(fit, target, sample_weight)
        return model


def apply(
    model: Classifier, features: Arrlike, h: int | None = None, w: int | None = None
) -> tuple[UInt8Arr, Arr]:
    is_2D = len(features.shape) == 3
    if is_2D:
        h, w, n_feats = features.shape
        flat_features = features.reshape((h * w, n_feats))
    else:
        assert h is not None
        assert w is not None
        flat_features = features
    probs: Arr = model.predict_proba(flat_features)
    classes = np.argmax(probs, axis=-1).astype(np.uint8)
    return classes.reshape((h, w)), probs.reshape((h, w, -1))


# TODO: once AC working, split into _featurise and featurise, the latter having an option to AC featurise
# if set by cfg, and if not jut calls _featurise
def featurise(image: Arr, feature_cfg: FeatureConfig, use_gpu: bool = False) -> Arrlike:
    if use_gpu:
        tensor = prepare_for_gpu(image)
        feats = multiscale_features_gpu(tensor, feature_cfg, tensor.dtype)
    else:
        feats = multiscale_features(image, feature_cfg)
    return feats


def train_and_apply(
    features: Arrlike, labels: UInt8Arr, train_cfg: TrainingConfig
) -> tuple[UInt8Arr, Arr, Classifier]:
    fit, target = get_labelled_training_data_from_stack(features, labels)
    fit, target = shuffle_sample_training_data(
        fit, target, train_cfg.shuffle_data, train_cfg.n_samples
    )
    model = get_model(
        train_cfg.classifier, train_cfg.classifier_params, train_cfg.use_gpu
    )
    model = train(model, fit, target, None)
    pred, probs = apply(model, features)
    return pred, probs, model
