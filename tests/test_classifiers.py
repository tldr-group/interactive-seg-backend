import pytest

from os import makedirs

import numpy as np

from interactive_seg_backend.configs import Arr, NPUIntArray, TrainingConfig, FeatureConfig
from test_core import e2e_get_miou

feat_cfg = FeatureConfig(add_weka_sigma_multiplier=False)

makedirs("tests/out", exist_ok=True)


def test_linear(feature_stack: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    tc = TrainingConfig(feature_config=feat_cfg, classifier="linear_regression")
    e2e_get_miou(
        feature_stack,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_linear.tif",
        True,
        0.4,
    )


def test_logistic(feature_stack: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    tc = TrainingConfig(feature_config=feat_cfg, classifier="logistic_regression", classifier_params={"max_iter": 1000})
    e2e_get_miou(
        feature_stack,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_logistic.tif",
        True,
        0.4,
    )


def test_xgb_cpu(feature_stack: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    tc = TrainingConfig(feature_config=feat_cfg, classifier="xgb")
    e2e_get_miou(
        feature_stack,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_xgbcpus.tif",
        True,
        0.5,
    )


MAX_ITERS = 3000


def test_mlp(feature_stack: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    tc = TrainingConfig(
        feature_config=feat_cfg,
        classifier="mlp",
        classifier_params={
            "hidden_layer_sizes": (50, 50, 50),
            "activation": "relu",
            "solver": "adam",
            "max_iter": MAX_ITERS,
            "warm_start": False,
            "random_state": 42,
        },
    )
    e2e_get_miou(
        feature_stack,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_mlp.tif",
        True,
        0.5,
    )


def test_otsu(feature_stack: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    tc = TrainingConfig(feature_config=feat_cfg, classifier="otsu")
    e2e_get_miou(
        feature_stack,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_otsu.tif",
        True,
        0.3,
    )


def test_kmeans(feature_stack: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    tc = TrainingConfig(
        feature_config=feat_cfg, classifier="kmeans", classifier_params={"scale": True, "random_state": 42}
    )
    e2e_get_miou(
        feature_stack,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_kmeans.tif",
        True,
        0.3,
    )


def test_seeded_region_growing(image: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    image_feature = image[:, :, np.newaxis]
    tc = TrainingConfig(feature_config=feat_cfg, classifier="seeded_region_growing", classifier_params={"scale": True})
    e2e_get_miou(
        image_feature,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_srg.tif",
        True,
        0.2,
    )


def test_srm(image: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    image_feature = image[:, :, np.newaxis]
    tc = TrainingConfig(feature_config=feat_cfg, classifier="srm", classifier_params={"scale": True, "Q": 16.0})
    e2e_get_miou(
        image_feature,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_srm.tif",
        True,
        0.3,
    )


def test_watershed(image: Arr, labels: NPUIntArray, ground_truth: NPUIntArray):
    image_feature = image[:, :, np.newaxis]
    tc = TrainingConfig(feature_config=feat_cfg, classifier="watershed", classifier_params={"use_gradient": True})
    e2e_get_miou(
        image_feature,
        labels,
        tc,
        ground_truth,
        True,
        "tests/out/0_seg_watershed.tif",
        True,
        0.2,
    )


if __name__ == "__main__":
    pytest.main(args=["-k test_classifiers", "-s"])
