import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, TrainingConfig, FeatureConfig
from test_core import test_e2e

feat_cfg = FeatureConfig(add_weka_sigma_multiplier=False)


@pytest.fixture
def train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    # need this as a placeholder s.t test e2e doesn't get mad
    # TODO: rewrite test_e2e and wrap it in a pytest function s.t it can be reused more robustly here
    extra_args = {"n_estimators": 200, "max_features": 2, "max_depth": None}
    return TrainingConfig(feat_cfg, n_samples=10000, classifier_params=extra_args)


def test_linear(feature_stack: Arr, labels: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="linear_regression")
    test_e2e(feature_stack, labels, tc, "tests/out/0_seg_linear.tif")


def test_logistic(feature_stack: Arr, labels: UInt8Arr):
    tc = TrainingConfig(feat_cfg, classifier="logistic_regression")
    test_e2e(feature_stack, labels, tc, "tests/out/0_seg_logistic.tif")


if __name__ == "__main__":
    pytest.main(args=["-k test_classifiers"])
