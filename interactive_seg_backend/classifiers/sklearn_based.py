from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier

from interactive_seg_backend.configs import NPFloatArray, NPUIntArray
from .base import Classifier

from typing import Any


class SklearnBasedClassifier(Classifier):
    """Wrapper for sklearn-based classifiers. Should be used for any sklearn-based classifier that implements fit and predict_proba methods."""

    def __init__(self, extra_args: dict[str, Any]) -> None:
        # self.model: Any

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ):
        """Generic fit for sklearn-API conforming models

        Args:
            train_data (NPFloatArray): (N_samples, C) array of training features.
            target_data (NPUIntArray): (N_samples,) array of integer class labels for training.
            sample_weights (NPFloatArray | None, optional): (N_samples,) array of optional sample weights. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        self.model.fit(train_data, target_data, sample_weight=sample_weights)
        return self

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        return self.model.predict_proba(features_flat)

    def predict(self, features: NPFloatArray):
        return super().predict(features)


class RandomForest(SklearnBasedClassifier):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = RandomForestClassifier(**extra_args)

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ):
        self.model.fit(train_data, target_data, sample_weight=sample_weights)
        return self

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        return self.model.predict_proba(features_flat)

    def predict(self, features: NPFloatArray):
        return super().predict(features)


class Logistic(SklearnBasedClassifier):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = LogisticRegression(**extra_args)


class Linear(Classifier):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = RidgeClassifier(**extra_args)

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ):
        self.model.fit(train_data, target_data, sample_weight=sample_weights)
        return self

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        return self.model.decision_function(features_flat)


class MLP(RandomForest):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = MLPClassifier(**extra_args)

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ):
        self.model.fit(train_data, target_data)
        return self
