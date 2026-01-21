import pytest
from tifffile import imwrite

from interactive_seg_backend.configs import Arr, UInt8Arr, FeatureConfig, TrainingConfig
from interactive_seg_backend.file_handling import (
    load_image,
    load_labels,
)
from interactive_seg_backend.features import multiscale_features
from interactive_seg_backend.processing import preprocess, postprocess
from interactive_seg_backend.utils import class_avg_miou

from test_core import e2e_get_miou

image = load_image("tests/data/1.tif")
labels = load_labels("tests/data/1_labels.tif")
ground_truth_ = load_labels("tests/data/1_ground_truth.tif")

feat_cfg = FeatureConfig(
    add_weka_sigma_multiplier=True,
)
extra_args = {"n_estimators": 200, "max_features": 2, "max_depth": None}
train_cfg = TrainingConfig(feature_config=feat_cfg, n_samples=10000, classifier_params=extra_args)


def test_equalize(save: bool = True) -> None:
    # test that perf for an image with greyscale intensity gradient better after histogram equalization
    feats_no_eq = multiscale_features(image, feat_cfg)

    equalized_img = preprocess(image, ("equalize",))
    feats_eq = multiscale_features(equalized_img, feat_cfg)

    if save:
        imwrite("tests/out/1_equalize_processed.tif", equalized_img)

    miou_no_eq, _ = e2e_get_miou(
        feats_no_eq,
        labels,
        train_cfg,
        ground_truth_,
        fname="tests/out/1_no_equalise.tif",
    )

    miou_eq, _ = e2e_get_miou(feats_eq, labels, train_cfg, ground_truth_, fname="tests/out/1_equalise.tif")
    assert miou_eq > miou_no_eq


def test_denoise(save: bool = True) -> None:
    denoised_img = preprocess(image, ("denoise",))
    feats_denoised = multiscale_features(denoised_img, feat_cfg)

    if save:
        imwrite("tests/out/1_denoised_processed.tif", denoised_img)

    e2e_get_miou(
        feats_denoised,
        labels,
        train_cfg,
        ground_truth_,
        save=save,
        fname="tests/out/1_denoised.tif",
    )


def test_modal(
    feature_stack: Arr,
    labels: UInt8Arr,
    ground_truth: UInt8Arr,
    out_fname: str = "tests/out/0_seg.tif",
):
    # we are modal filtering a seg with small tertiary phase -> mIoU should decrease
    miou, pred = e2e_get_miou(feature_stack, labels, train_cfg, ground_truth, save=False, run_checks=False)
    filtered = postprocess(pred, ("modal_filter",))
    filtered_miou = class_avg_miou(filtered, ground_truth)
    assert filtered_miou < miou


if __name__ == "__main__":
    pytest.main(args=["-k test_processing", "-s"])
