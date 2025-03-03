import pytest

from interactive_seg_backend.configs import Arr, UInt8Arr, TrainingConfig, FeatureConfig
from interactive_seg_backend.file_handling import (
    save_segmentation,
    load_image,
    load_labels,
)
from interactive_seg_backend.core import train_and_apply
from interactive_seg_backend.extensions import (
    autocontext_features,
    CRFParams,
    do_crf_from_probabilites,
)


@pytest.fixture
def train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    extra_args = {"n_estimators": 200, "max_features": 2, "max_depth": None}
    return TrainingConfig(feat_cfg, n_samples=10000, classifier_params=extra_args)


image_ = load_image("tests/data/1.tif")
labels_ = load_labels("tests/data/1_labels.tif")


def test_autocontext_features(
    train_cfg: TrainingConfig,
) -> None:
    af_feats = autocontext_features(
        image_,
        labels_,
        train_cfg,
    )
    pred, _, _ = train_and_apply(af_feats, labels_, train_cfg)
    save_segmentation(pred, "tests/out/1_seg_autocontext.tif")


def test_crf(
    image: Arr,
    feature_stack: Arr,
    labels: UInt8Arr,
    train_cfg: TrainingConfig,
    ground_truth: UInt8Arr,
    out_fname: str = "tests/out/0_seg.tif",
):
    _, probs, _ = train_and_apply(feature_stack, labels, train_cfg)
    params = CRFParams()
    improved = do_crf_from_probabilites(probs, image, 3, params)

    save_segmentation(improved, "tests/out/0_seg_crf.tif")


if __name__ == "__main__":
    pytest.main(args=["-k test_extensions", "-s"])
