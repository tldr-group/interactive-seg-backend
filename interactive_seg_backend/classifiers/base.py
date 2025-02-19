import numpy as np
from pickle import load, dump
from skops.io import load as skload, dump as skdump


from abc import ABCMeta, abstractmethod
from typing import Any

from interactive_seg_backend.configs import Arr, UInt8Arr


class Classifier(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, extra_args: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fit(self, train_data: Arr, target_data: UInt8Arr):
        return self

    @abstractmethod
    def predict_proba(self, features_flat: Arr) -> Arr:
        raise NotImplementedError

    # TODO: make these polymorphic?
    @abstractmethod
    def predict(self, features: Arr) -> UInt8Arr:
        h, w, c = features.shape
        features_flat = features.reshape((h * w, c))
        probs = self.predict_proba(features_flat)
        seg_flat = np.argmax(probs, axis=-1)
        return seg_flat.reshape((h, w))

    @abstractmethod
    def save(self, out_path: str, as_skops: bool = False) -> None:
        if as_skops is False:
            with open(out_path, "wb") as f:
                dump(self, f)
        else:
            skdump(self, out_path)


def load_classifier(path: str) -> Classifier:
    obj: Classifier
    ext = path.split(".")[-1]
    if "pkl" in ext:
        with open(path, "rb") as f:
            obj = load(f)
    elif "skops" in ext:
        obj = skload(path)
    else:
        raise NotImplementedError
    return obj
