import numpy as np
from sklearn.ensemble import RandomForestClassifier

from interactive_seg_backend.configs import Arr, UInt8Arr
from .base import Classifier

from typing import Any


class RandomForest(Classifier):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = RandomForestClassifier(**extra_args)

    def fit(self, train_data: Arr, target_data: UInt8Arr):
        self.model.fit(train_data, target_data)
        return self

    def predict_proba(self, features_flat: Arr) -> Arr:
        return self.model.predict_proba(features_flat)
