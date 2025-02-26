import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, TrainingConfig, FeatureConfig
from test_core import e2e_get_miou

feat_cfg = FeatureConfig(add_weka_sigma_multiplier=False)


def test_linear(feature_stack: Arr, labels: UInt8Arr, ground_truth: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="linear_regression")
    e2e_get_miou(
        feature_stack, labels, tc, ground_truth, True, "tests/out/0_seg_linear.tif"
    )


def test_logistic(feature_stack: Arr, labels: UInt8Arr, ground_truth: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="logistic_regression")
    e2e_get_miou(
        feature_stack, labels, tc, ground_truth, True, "tests/out/0_seg_logistic.tif"
    )


def test_xgb_cpu(feature_stack: Arr, labels: UInt8Arr, ground_truth: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="xgb")
    e2e_get_miou(
        feature_stack, labels, tc, ground_truth, True, "tests/out/0_seg_xgbcpus.tif"
    )


if __name__ == "__main__":
    pytest.main(args=["-k test_classifiers", "-s"])
