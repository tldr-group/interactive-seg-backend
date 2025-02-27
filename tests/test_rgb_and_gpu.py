import pytest
from tifffile import imread
import numpy as np
import torch

from interactive_seg_backend.configs import Arr, UInt8Arr, FeatureConfig, TrainingConfig
from interactive_seg_backend.features import (
    multiscale_features,
    prepare_for_gpu,
    multiscale_features_gpu,
)


img = imread("tests/data/rgb.tif")

feat_cfg = FeatureConfig(
    add_weka_sigma_multiplier=True, max_sigma=8, difference_of_gaussians=False
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


def test_gpu_featurise() -> None:
    rgb_img_tensor = prepare_for_gpu(img, "cuda:0")
    feats_rgb = multiscale_features_gpu(rgb_img_tensor, feat_cfg, torch.float32)

    r = img[:, :, 0]
    img_tensor = prepare_for_gpu(r, "cuda:0")
    feats_greyscale = multiscale_features_gpu(img_tensor, feat_cfg, torch.float32)

    assert feats_greyscale.shape[-1] == feats_rgb.shape[-1] // 3


if __name__ == "__main__":
    pytest.main(args=["-k test_rgb_gpu", "-s"])
