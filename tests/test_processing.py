import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, FeatureConfig, TrainingConfig
from interactive_seg_backend.file_handling import (
    load_image,
    load_labels,
)
from interactive_seg_backend.features import multiscale_features
from interactive_seg_backend.processing import preprocess, postprocess

from test_core import e2e_get_miou

image = load_image("tests/data/1.tif")
labels = load_labels("tests/data/1_labels.tif")
ground_truth_ = load_labels("tests/data/1_ground_truth.tif")

feat_cfg = FeatureConfig(
    add_weka_sigma_multiplier=True,
)
extra_args = {"n_estimators": 200, "max_features": 2, "max_depth": None}
train_cfg = TrainingConfig(feat_cfg, n_samples=10000, classifier_params=extra_args)


def test_equalize() -> None:
    feats_no_eq = multiscale_features(image, feat_cfg)

    equalized_img = preprocess(image, ("equalize",))
    feats_eq = multiscale_features(equalized_img, feat_cfg)

    miou_no_eq = e2e_get_miou(
        feats_no_eq,
        labels,
        train_cfg,
        ground_truth_,
        fname="tests/out/1_no_equalise.tif",
    )

    miou_eq = e2e_get_miou(
        feats_eq, labels, train_cfg, ground_truth_, fname="tests/out/1_equalise.tif"
    )
    assert miou_eq > miou_no_eq


if __name__ == "__main__":
    pytest.main(args=["-k test_processing", "-s"])
