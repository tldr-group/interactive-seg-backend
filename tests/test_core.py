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


def test_e2e(
    feature_stack: Arr,
    labels: UInt8Arr,
    train_cfg: TrainingConfig,
    out_fname: str = "tests/out/0_seg.tif",
):
    fit, target = get_labelled_training_data_from_stack(feature_stack, labels)
    fit, target = shuffle_sample_training_data(
        fit, target, train_cfg.shuffle_data, train_cfg.n_samples
    )
    model = get_model(train_cfg.classifier, train_cfg.classifier_params)
    model = train(model, fit, target, None)
    pred = apply(model, feature_stack)
    rh, rw = pred.shape
    fh, fw, _ = feature_stack.shape

    save_segmentation(pred, out_fname)

    assert rh == fh
    assert rw == fw

    ground_truth = load_labels("tests/data/0_ground_truth.tif")
    miou = class_avg_miou(pred, ground_truth)
    assert miou > MIOU_CUTOFF


if __name__ == "__main__":
    pytest.main(args=["-k test_core"])
