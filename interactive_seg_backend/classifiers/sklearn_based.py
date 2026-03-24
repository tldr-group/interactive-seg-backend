from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier

from interactive_seg_backend.configs import NPFloatArray, NPUIntArray
from .base import Classifier

from typing import Any


class RandomForest(Classifier):
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


class Logistic(RandomForest):
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
