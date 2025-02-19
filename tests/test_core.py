import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, FeatureConfig, TrainingConfig
from interactive_seg_backend.features import multiscale_features
from interactive_seg_backend.file_handling import load_image, load_labels
from interactive_seg_backend.core import (
    get_labelled_training_data_from_stack,
    shuffle_sample_training_data,
    get_model,
    train,
    apply,
)


TEST_IMAGE_SHAPE = (512, 512)


@pytest.fixture
def image() -> Arr:
    return load_image("tests/data/0.tif")


@pytest.fixture
def labels() -> Arr:
    return load_labels("tests/data/0_labels.tif")


def test_load_image(image: Arr) -> None:
    assert image.shape == TEST_IMAGE_SHAPE


def test_load_labels(labels: UInt8Arr) -> None:
    assert labels.shape == TEST_IMAGE_SHAPE


@pytest.fixture
def feat_cfg() -> FeatureConfig:
    return FeatureConfig(add_weka_sigma_multiplier=False)


@pytest.fixture
def feature_stack(image: Arr, feat_cfg: FeatureConfig) -> Arr:
    return multiscale_features(image, feat_cfg)


def test_feat_stack(feat_cfg: FeatureConfig, feature_stack: Arr):
    _, _, c = feature_stack.shape
    n_feats_from_config = len(feat_cfg.get_filter_strings())
    assert n_feats_from_config == c


@pytest.fixture
def train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    extra_args = {"n_estimators": 200, "max_features": 2, "max_depth": None}
    return TrainingConfig(feat_cfg, n_samples=10000, classifier_params=extra_args)


def test_e2e(feature_stack: Arr, labels: UInt8Arr, train_cfg: TrainingConfig):
    fit, target = get_labelled_training_data_from_stack(feature_stack, labels)
    fit, target = shuffle_sample_training_data(
        fit, target, train_cfg.shuffle_data, train_cfg.n_samples
    )
    model = get_model(train_cfg.classifier, train_cfg.classifier_params)
    model = train(model, fit, target, None)
    res = apply(model, feature_stack)
    rh, rw = res.shape
    fh, fw, _ = feature_stack.shape
    assert rh == fh
    assert rw == fw


if __name__ == "__main__":
    pytest.main(args=["-k test_core"])
