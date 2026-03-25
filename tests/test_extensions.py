import pytest

import numpy as np
from skimage.measure import label
from interactive_seg_backend.configs import Arr, NPUIntArray, TrainingConfig, FeatureConfig, NPFloatArray, ClassInfo
from interactive_seg_backend.file_handling import (
    save_segmentation,
    load_image,
    load_labels,
)
from interactive_seg_backend.extensions import (
    autocontext_features,
    CRF_AVAILABLE,
    CRFParams,
    do_crf_from_probabilites,
    ExpertSegClassifier,
)
from interactive_seg_backend import train_and_apply, transfer_from_gpu
from interactive_seg_backend.features import multiscale_features


@pytest.fixture
def train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    extra_args: dict[str, int | str | None] = {
        "n_estimators": 200,
        "max_features": 2,
        "max_depth": None,
    }
    return TrainingConfig(feature_config=feat_cfg, n_samples=10000, classifier_params=extra_args)


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
    af_feats = transfer_from_gpu(af_feats)
    pred, _, _ = train_and_apply(af_feats, labels_, train_cfg)
    save_segmentation(pred, "tests/out/1_seg_autocontext.tif")


@pytest.mark.skipif(not CRF_AVAILABLE, reason="requires CRF be installed")
def test_crf(
    image: Arr,
    feature_stack: NPFloatArray,
    labels: NPUIntArray,
    train_cfg: TrainingConfig,
    ground_truth: NPUIntArray,
    out_fname: str = "tests/out/0_seg.tif",
):
    _, probs, _ = train_and_apply(feature_stack, labels, train_cfg)
    params = CRFParams()
    improved = do_crf_from_probabilites(probs, image, 3, params)

    save_segmentation(improved, "tests/out/0_seg_crf.tif")


@pytest.fixture
def es_image() -> NPFloatArray:
    return load_image("tests/data/3.tif")


@pytest.fixture
def es_labels() -> NPUIntArray:
    return load_labels("tests/data/3_labels.tif")


@pytest.fixture
def es_features(es_image: NPFloatArray, feat_cfg: FeatureConfig) -> NPFloatArray:
    return multiscale_features(es_image, feat_cfg)


@pytest.fixture
def es_vf_train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    class_infos = [
        ClassInfo(name="solid", value=0, desired_volume_fraction=0.4),
        ClassInfo(name="pore", value=1, desired_volume_fraction=None),
    ]
    return TrainingConfig(feature_config=feat_cfg, class_infos=class_infos)


def test_expertseg_vf(es_features: NPFloatArray, es_labels: NPUIntArray, es_vf_train_cfg: TrainingConfig) -> None:
    class_infos = es_vf_train_cfg.class_infos
    model = ExpertSegClassifier(
        class_infos=class_infos, n_epochs=50, lambd_vf=2, extra_args={"max_depth": 6, "eta": 0.1}
    )
    model.fit(es_features, es_labels)
    preds = model.predict(es_features) - 1

    solid_vf = class_infos[0].desired_volume_fraction
    assert solid_vf is not None

    pred_solid_vf = float(np.mean(np.where(preds == class_infos[0].value, 1, 0)))
    assert np.isclose(pred_solid_vf, solid_vf, rtol=0.05)

    save_segmentation(preds, "tests/out/es_vf_seg.tif")


@pytest.fixture
def es_conn_train_cfg(feat_cfg: FeatureConfig) -> TrainingConfig:
    class_infos = [
        ClassInfo(name="solid", value=0, connectivity_target="minimise", desired_volume_fraction=0.52),
        ClassInfo(name="pore", value=1),
    ]
    return TrainingConfig(
        feature_config=feat_cfg,
        class_infos=class_infos,
        classifier="xgb",
        classifier_params={"max_depth": 6, "eta": 0.1},
    )


def test_expertseg_conn(es_features: NPFloatArray, es_labels: NPUIntArray, es_conn_train_cfg: TrainingConfig) -> None:
    class_infos = es_conn_train_cfg.class_infos

    es_conn_model = ExpertSegClassifier(
        class_infos=class_infos, n_epochs=100, lambd_conn=3, extra_args={"max_depth": 6, "eta": 0.1}
    )
    es_conn_model.fit(es_features, es_labels)
    es_conn_preds = es_conn_model.predict(es_features) - 1
    _, n_ccs_es_conn = label(es_conn_preds, background=class_infos[1].value, return_num=True)

    es_vf_model = es_conn_model
    es_vf_model.do_conn_loss = False
    es_vf_model.fit(es_features, es_labels)
    es_vfs_preds = es_conn_model.predict(es_features) - 1
    _, n_ccs_es_vf = label(es_vfs_preds, background=class_infos[1].value, return_num=True)

    save_segmentation(es_conn_preds, "tests/out/es_conn_vf_seg.tif")
    save_segmentation(es_vfs_preds, "tests/out/es_vf_seg_.tif")
    assert n_ccs_es_conn > n_ccs_es_vf


if __name__ == "__main__":
    pytest.main(args=["-k test_extensions", "-s"])
