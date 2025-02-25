import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, TrainingConfig, FeatureConfig
from test_core import e2e

feat_cfg = FeatureConfig(add_weka_sigma_multiplier=False)


def test_linear(feature_stack: Arr, labels: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="linear_regression")
    e2e(feature_stack, labels, tc, True, "tests/out/0_seg_linear.tif")


def test_logistic(feature_stack: Arr, labels: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="logistic_regression")
    e2e(feature_stack, labels, tc, True, "tests/out/0_seg_logistic.tif")


if __name__ == "__main__":
    pytest.main(args=["-k test_classifiers"])
