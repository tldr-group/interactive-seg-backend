from xgboost import XGBClassifier
import numpy as np

from interactive_seg_backend.configs import Arr, UInt8Arr
from interactive_seg_backend.classifiers import RandomForest

from typing import Any


class XGBCPU(RandomForest):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = XGBClassifier(**extra_args)

    def fit(
        self, train_data: Arr, target_data: UInt8Arr, sample_weights: Arr | None = None
    ):
        self.model.fit(train_data, target_data - 1, sample_weight=sample_weights)
        return self

    def predict(self, features: Arr) -> UInt8Arr:
        arr = super().predict(features)
        return np.add(arr, 1)
