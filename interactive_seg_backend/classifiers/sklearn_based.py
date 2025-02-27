from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from interactive_seg_backend.configs import Arr, Arrlike, UInt8Arr
from .base import Classifier

from typing import Any


class RandomForest(Classifier):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = RandomForestClassifier(**extra_args)

    def fit(
        self,
        train_data: Arrlike,
        target_data: UInt8Arr,
        sample_weights: Arr | None = None,
    ):
        self.model.fit(train_data, target_data, sample_weight=sample_weights)
        return self

    def predict_proba(self, features_flat: Arrlike) -> Arr:
        return self.model.predict_proba(features_flat)

    def predict(self, features: Arrlike):
        return super().predict(features)


class Logistic(RandomForest):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = LogisticRegression(**extra_args)


class Linear(RandomForest):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = RidgeClassifier(**extra_args)

    def predict_proba(self, features_flat: Arr) -> Arr:
        return self.model.decision_function(features_flat)
