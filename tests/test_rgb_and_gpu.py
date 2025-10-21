import pytest
from tifffile import imread
import numpy as np

from interactive_seg_backend.configs import Arr, UInt8Arr, FeatureConfig, TrainingConfig
from interactive_seg_backend.features import (
    multiscale_features,
    TORCH_AVAILABLE,
    prepare_for_gpu,
    multiscale_features_gpu,
)

from test_core import e2e_get_miou


img = imread("tests/data/rgb.tif")

feat_cfg = FeatureConfig(
    add_weka_sigma_multiplier=True,
    max_sigma=8,
    difference_of_gaussians=False,
    use_gpu=True,
)


@pytest.fixture
def train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    # extra_args = {"n_estimators": 200, "max_features": 2, "max_depth": None}
    return TrainingConfig(
        feat_cfg, classifier="xgb", n_samples=10000, classifier_params={}, use_gpu=True
    )


def test_rgb() -> None:
    feats_r = multiscale_features(img[:, :, 0], feat_cfg)
    feats_rgb = multiscale_features(img, feat_cfg)

    assert feats_r.shape[-1] == feats_rgb.shape[-1] // 3


def test_annoying_gresycale() -> None:
    r = img[:, :, 0]
    expanded = np.expand_dims(img[:, :, 0], -1)

    feats_r = multiscale_features(r, feat_cfg)
    feats_e = multiscale_features(expanded, feat_cfg)

    assert feats_r.shape == feats_e.shape

@pytest.mark.skipif(not TORCH_AVAILABLE)
def test_gpu_featurise() -> None:
    rgb_img_tensor = prepare_for_gpu(img, "cuda:0")
    feats_rgb = multiscale_features_gpu(rgb_img_tensor, feat_cfg)

    r = img[:, :, 0]
    img_tensor = prepare_for_gpu(r, "cuda:0")
    feats_greyscale = multiscale_features_gpu(img_tensor, feat_cfg)

    assert feats_greyscale.shape[-1] == feats_rgb.shape[-1] // 3

@pytest.mark.skipif(not TORCH_AVAILABLE)
def test_gpu_e2e(
    image: Arr, labels: UInt8Arr, train_cfg: TrainingConfig, ground_truth: UInt8Arr
) -> None:
    img_tensor = prepare_for_gpu(image)
    feats = multiscale_features_gpu(img_tensor, feat_cfg)
    e2e_get_miou(
        feats, labels, train_cfg, ground_truth, fname="tests/out/0_seg_gpu.tif"
    )


if __name__ == "__main__":
    pytest.main(args=["-k test_rgb_and_gpu", "-s"])
