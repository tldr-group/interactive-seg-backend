import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, TrainingConfig, FeatureConfig
from interactive_seg_backend.file_handling import (
    save_segmentation,
    load_image,
    load_labels,
)
from interactive_seg_backend.core import train_and_apply
from interactive_seg_backend.extensions import autocontext_features


@pytest.fixture
def train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    extra_args = {"n_estimators": 200, "max_features": 2, "max_depth": None}
    return TrainingConfig(feat_cfg, n_samples=10000, classifier_params=extra_args)


image = load_image("tests/data/1.tif")
labels = load_labels("tests/data/1_labels.tif")


def test_autocontext_features(
    train_cfg: TrainingConfig,
) -> None:
    af_feats = autocontext_features(
        image,
        labels,
        train_cfg,
    )
    pred, _, _ = train_and_apply(af_feats, labels, train_cfg)
    save_segmentation(pred, "tests/out/1_seg_autocontext.tif")


if __name__ == "__main__":
    pytest.main(args=["-k test_extensions", "-s"])
