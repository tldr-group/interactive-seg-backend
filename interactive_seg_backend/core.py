import numpy as np
from sklearn.ensemble import RandomForestClassifier  # TODO: generalise to classifiers

from interactive_seg_backend.configs.types import Arr, UInt8Arr, ClassifierNames
from interactive_seg_backend.classifiers import Classifier, RandomForest

from typing import Any

# Classifier: TypeAlias = RandomForestClassifier


def get_labelled_training_data_from_stack(
    feature_stack: Arr, labels: UInt8Arr
) -> tuple[Arr, UInt8Arr]:
    h, w, n_feats = feature_stack.shape
    flat_labels = labels.reshape((h * w))
    flat_features = feature_stack.reshape((h * w, n_feats))
    labelled_mask = np.nonzero(flat_labels)

    fit_data = flat_features[labelled_mask[0], :]
    target_data = flat_labels[labelled_mask[0]]

    return fit_data, target_data


def shuffle_sample_training_data(
    fit: Arr, target: UInt8Arr, shuffle: bool = True, sample_n: int = -1
) -> tuple[Arr, UInt8Arr]:
    n_samples = target.shape[0]
    all_inds = np.arange(0, n_samples, 1)
    if shuffle:
        np.random.shuffle(all_inds)
    if sample_n > 0:
        sample_inds = all_inds[:sample_n]
        return fit[sample_inds], target[sample_inds]
    else:
        return fit[all_inds], target[all_inds]


def get_model(model_type: ClassifierNames, extra_args: dict[str, Any]) -> Classifier:
    if model_type == "random_forest":
        return RandomForest(extra_args)
    else:
        raise Exception("Not implemented!")


def train(
    model: Classifier, fit: Arr, target: UInt8Arr, sample_weight: Arr | None
) -> Classifier:
    if sample_weight is None:
        model.fit(fit, target)
        return model
    else:
        model.fit(fit, target, sample_weight)
        return model


def apply(
    model: Classifier, features: Arr, h: int | None = None, w: int | None = None
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
