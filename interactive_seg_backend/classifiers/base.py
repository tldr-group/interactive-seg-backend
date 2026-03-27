import numpy as np
from pickle import load, dump
from skops.io import load as skload, dump as skdump


from abc import ABC
from typing import Any

from interactive_seg_backend.configs import NPFloatArray, NPUIntArray


class Classifier(ABC):
    """Abstract base class interface for classifiers."""

    def __init__(self, extra_args: dict[str, Any]) -> None:
        pass

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ):
        raise NotImplementedError

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        raise NotImplementedError

    # Assuming all GPU models return their probs as numpy arr this frame should work
    def predict(self, features: NPFloatArray) -> NPUIntArray:
        h, w, c = features.shape
        features_flat = features.reshape((h * w, c))
        probs = self.predict_proba(features_flat)
        seg_flat = np.argmax(probs, axis=-1)
        return seg_flat.reshape((h, w))

    def save(self, out_path: str, as_skops: bool = False) -> None:
        if as_skops is False:
            with open(out_path, "wb") as f:
                dump(self, f)
        else:
            skdump(self, out_path)

    def __repr__(self) -> str:
        name: str
        model = getattr(self, "model", None)
        if model is not None:
            name = model.__class__.__name__
        else:
            name = "None"
        return f"{name}"


def load_classifier(path: str) -> Classifier:
    """Simpler helper to load classifier objects saved in either pickle or skops format."""
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
