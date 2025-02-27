from xgboost import XGBClassifier, DMatrix
import numpy as np

from interactive_seg_backend.configs import Arr, UInt8Arr, Arrlike
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


class XGBGPU(RandomForest):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        extra_args["device"] = "cuda"
        extra_args["tree_method"] = "hist"
        self.model = XGBClassifier(**extra_args)

    def fit(
        self,
        train_data: Arrlike,
        target_data: UInt8Arr,
        sample_weights: Arr | None = None,
    ):
        data = DMatrix(data=train_data, label=target_data - 1)
        self.model.fit(data.get_data(), data.get_label())
        return self

    def predict_proba(self, features_flat: Arr) -> Arr:
        return self.model.predict_proba(features_flat)

    def predict(self, features: Arrlike) -> UInt8Arr:
        h, w, c = features.shape
        features_flat = features.reshape((h * w, c))
        data = DMatrix(features_flat)
        probs = self.predict_proba(data)
        seg_flat = np.argmax(probs, axis=-1)
        return seg_flat.reshape((h, w))
