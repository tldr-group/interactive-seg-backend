import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, FeatureConfig, TrainingConfig
from interactive_seg_backend.file_handling import (
    load_labels,
    save_segmentation,
)
from interactive_seg_backend.core import (
    get_labelled_training_data_from_stack,
    shuffle_sample_training_data,
    get_model,
    train,
    apply,
)
from interactive_seg_backend.utils import class_avg_miou


TEST_IMAGE_SHAPE = (512, 512)


def test_load_image(image: Arr) -> None:
    assert image.shape == TEST_IMAGE_SHAPE


def test_load_labels(labels: UInt8Arr) -> None:
    assert labels.shape == TEST_IMAGE_SHAPE


def test_feat_stack(feat_cfg: FeatureConfig, feature_stack: Arr):
    _, _, c = feature_stack.shape
    n_feats_from_config = len(feat_cfg.get_filter_strings())
    assert n_feats_from_config == c


@pytest.fixture
def train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    extra_args = {"n_estimators": 200, "max_features": 2, "max_depth": None}
    return TrainingConfig(feat_cfg, n_samples=10000, classifier_params=extra_args)


MIOU_CUTOFF = 0.8


def e2e(
    features: Arr,
    label: UInt8Arr,
    cfg: TrainingConfig,
    save: bool = True,
    fname: str = "tests/out/0_seg.tif",
):
    fit, target = get_labelled_training_data_from_stack(features, label)
    fit, target = shuffle_sample_training_data(
        fit, target, cfg.shuffle_data, cfg.n_samples
    )
    model = get_model(cfg.classifier, cfg.classifier_params)
    model = train(model, fit, target, None)
    pred = apply(model, features)
    rh, rw = pred.shape
    fh, fw, _ = features.shape

    if save:
        save_segmentation(pred, fname)
    assert rh == fh
    assert rw == fw

    ground_truth = load_labels("tests/data/0_ground_truth.tif")
    miou = class_avg_miou(pred, ground_truth)
    assert miou > MIOU_CUTOFF


def test_e2e(
    feature_stack: Arr,
    labels: UInt8Arr,
    train_cfg: TrainingConfig,
    out_fname: str = "tests/out/0_seg.tif",
):
    e2e(feature_stack, labels, train_cfg, True, out_fname)


if __name__ == "__main__":
    pytest.main(args=["-k test_core"])
