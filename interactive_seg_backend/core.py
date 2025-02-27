import numpy as np
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
) -> UInt8Arr:
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
    return classes.reshape((h, w))
