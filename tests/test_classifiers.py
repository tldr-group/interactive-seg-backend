import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, TrainingConfig, FeatureConfig
from test_core import e2e_get_miou

feat_cfg = FeatureConfig(add_weka_sigma_multiplier=False)


def test_linear(feature_stack: Arr, labels: UInt8Arr, ground_truth: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="linear_regression")
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


def test_logistic(feature_stack: Arr, labels: UInt8Arr, ground_truth: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="logistic_regression", classifier_params={"max_iter": 1000})
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


def test_xgb_cpu(feature_stack: Arr, labels: UInt8Arr, ground_truth: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="xgb")
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


def test_mlp(feature_stack: Arr, labels: UInt8Arr, ground_truth: UInt8Arr):
    tc = TrainingConfig(
        feat_cfg,
        classifier="mlp",
        classifier_params={
            "hidden_layer_sizes": (50, 50, 50),
            "activation": "relu",
            "solver": "adam",
            "max_iter": MAX_ITERS,
            "warm_start": False,
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


if __name__ == "__main__":
    pytest.main(args=["-k test_classifiers", "-s"])
