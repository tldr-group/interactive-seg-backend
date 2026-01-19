import numpy as np
from pickle import load, dump
from skops.io import load as skload, dump as skdump


from typing import Any

from interactive_seg_backend.configs import Arr, UInt8Arr, Arrlike


class Classifier(object):
    def __init__(self, extra_args: dict[str, Any]) -> None:
        self.model: object | None = None

    def fit(
        self,
        train_data: Arrlike,
        target_data: UInt8Arr,
        sample_weights: Arr | None = None,
    ):
        return self

    def predict_proba(self, features_flat: Arrlike) -> Arr:
        raise NotImplementedError

    # Assuming all GPU models return their probs as numpy arr this frame should work
    def predict(self, features: Arrlike) -> UInt8Arr:
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
        model_name = "None" if self.model is None else self.model.__class__.__name__
        return f"{model_name}"


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
