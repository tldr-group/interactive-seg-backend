from xgboost import XGBClassifier
import numpy as np

from interactive_seg_backend.configs import NPFloatArray, NPUIntArray
from interactive_seg_backend.features.gpu_utils import optionally_pass_to_cupy
from interactive_seg_backend.classifiers import RandomForest

from typing import Any


class XGBCPU(RandomForest):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model = XGBClassifier(**extra_args)

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ):
        self.model.fit(train_data, target_data - 1, sample_weight=sample_weights)
        return self

    def predict(self, features: NPFloatArray):
        arr = super().predict(features)
        return np.add(arr, 1)


class XGBGPU(RandomForest):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        extra_args["device"] = "gpu"
        extra_args["tree_method"] = "hist"
        self.model = XGBClassifier(**extra_args)

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ):
        train_data = optionally_pass_to_cupy(train_data)
        target_data = optionally_pass_to_cupy(target_data)
        self.model.fit(train_data, target_data - 1, sample_weight=sample_weights)
        return self

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        features_flat = optionally_pass_to_cupy(features_flat)
        return self.model.predict_proba(features_flat)

    def predict(self, features: NPFloatArray) -> NPUIntArray:
        h, w, c = features.shape
        features_flat = features.reshape((h * w, c))
        probs = self.predict_proba(features_flat)
        seg_flat = np.argmax(probs, axis=-1)
        return seg_flat.reshape((h, w))
