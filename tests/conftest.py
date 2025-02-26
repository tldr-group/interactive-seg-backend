import pytest
from interactive_seg_backend.configs import Arr, FeatureConfig
from interactive_seg_backend.features import multiscale_features
from interactive_seg_backend.file_handling import (
    load_image,
    load_labels,
)


@pytest.fixture(scope="module")
def image() -> Arr:
    return load_image("tests/data/0.tif")


@pytest.fixture(scope="module")
def labels() -> Arr:
    return load_labels("tests/data/0_labels.tif")


@pytest.fixture(scope="module")
def feat_cfg() -> FeatureConfig:
    return FeatureConfig(add_weka_sigma_multiplier=False)


@pytest.fixture(scope="module")
def feature_stack(image: Arr, feat_cfg: FeatureConfig) -> Arr:
    return multiscale_features(image, feat_cfg)


@pytest.fixture(scope="module")
def ground_truth() -> Arr:
    return load_labels("tests/data/0_ground_truth.tif")
