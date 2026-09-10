import pytest
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
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
    SAM_AVAILABLE,
    do_sam_postproc,
)
from interactive_seg_backend.extensions.autocontext import (
    compute_ray_autocontext_features,
)
from interactive_seg_backend.extensions.sam_onnx import (
    SAMEncoderONNX,
    SAMDecoderONNX,
    to_onnx_image,
    to_onnx_point_prompt,
    to_onnx_box_prompt,
    _get_default_cache_path_for_platform,
    load_or_download_model,
)
from interactive_seg_backend import train_and_apply, transfer_from_gpu, featurise_
from interactive_seg_backend.features import multiscale_features
from interactive_seg_backend.utils import class_avg_miou


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


def test_compute_ray_autocontext_features() -> None:
    # 5x5 grid with hot pixel at center (2, 2)
    grid = np.zeros((5, 5), dtype=np.float32)
    grid[2, 2] = 1.0

    # 4 rays (Up, Down, Left, Right) at distance 1
    rolled_4 = compute_ray_autocontext_features(grid, distances=[1], n_rays=4)
    assert rolled_4.shape == (5, 5, 4)
    # Up (-1, 0): rolled up, so pixel at (1, 2) has the value that was at (2, 2)
    assert rolled_4[1, 2, 0] == 1.0
    # Down (1, 0): rolled down, so pixel at (3, 2) has value from (2, 2)
    assert rolled_4[3, 2, 1] == 1.0
    # Left (0, -1): rolled left, so pixel at (2, 1) has value from (2, 2)
    assert rolled_4[2, 1, 2] == 1.0
    # Right (0, 1): rolled right, so pixel at (2, 3) has value from (2, 2)
    assert rolled_4[2, 3, 3] == 1.0

    # 8 rays at distance 1 includes diagonals
    rolled_8 = compute_ray_autocontext_features(grid, distances=[1], n_rays=8)
    assert rolled_8.shape == (5, 5, 8)
    # Up-Left (-1, -1)
    assert rolled_8[1, 1, 4] == 1.0
    # Up-Right (-1, 1)
    assert rolled_8[1, 3, 5] == 1.0
    # Down-Left (1, -1)
    assert rolled_8[3, 1, 6] == 1.0
    # Down-Right (1, 1)
    assert rolled_8[3, 3, 7] == 1.0

    # Single-pass multi-distance check (distances 1 and 2)
    rolled_dist = compute_ray_autocontext_features(grid, distances=[1, 2], n_rays=4)
    assert rolled_dist.shape == (5, 5, 8)  # 4 rays * 2 distances = 8 channels
    # Up at dist 1 -> channel 0, Up at dist 2 -> channel 1
    assert rolled_dist[1, 2, 0] == 1.0
    assert rolled_dist[0, 2, 1] == 1.0


def test_autocontext_features_all_modes(train_cfg: TrainingConfig) -> None:
    # 1. Simple mode: appends raw probabilities
    train_cfg_simple = train_cfg.model_copy(update={"autocontext_type": "simple"})
    simple_feats = autocontext_features(image_, labels_, train_cfg_simple)
    simple_feats = transfer_from_gpu(simple_feats)

    # Base features without autocontext
    base_train_cfg = train_cfg.model_copy(update={"autocontext_type": False})
    base_feats = featurise_(image_, base_train_cfg.feature_config)
    base_feats = transfer_from_gpu(base_feats)
    # Number of classes in labels_
    n_classes = len(np.unique(labels_[labels_ > 0]))
    print(f"Base features shape: {base_feats.shape}, Simple autocontext features shape: {simple_feats.shape}")
    assert simple_feats.shape[-1] == base_feats.shape[-1] + n_classes

    # 2. Original ray mode (4 rays, 3 distances)
    train_cfg_orig_4 = train_cfg.model_copy(
        update={"autocontext_type": "original", "autocontext_distances": [5, 50, 100], "autocontext_n_rays": 4}
    )
    orig_feats_4 = autocontext_features(image_, labels_, train_cfg_orig_4)
    orig_feats_4 = transfer_from_gpu(orig_feats_4)
    # Expected channels added: 4 rays * 3 distances * n_classes
    # Base feat channels = (simple_feats.shape[-1] - n_classes)
    n_base = base_feats.shape[-1]
    assert orig_feats_4.shape[-1] == n_base + 4 * 3 * n_classes

    # 3. Original ray mode (8 rays, 2 distances)
    train_cfg_orig_8 = train_cfg.model_copy(
        update={"autocontext_type": "original", "autocontext_distances": [5, 10], "autocontext_n_rays": 8}
    )
    orig_feats_8 = autocontext_features(image_, labels_, train_cfg_orig_8)
    orig_feats_8 = transfer_from_gpu(orig_feats_8)
    assert orig_feats_8.shape[-1] == n_base + 8 * 2 * n_classes

    # 4. Ilastik mode
    train_cfg_ilastik = train_cfg.model_copy(update={"autocontext_type": "ilastik"})
    ilastik_feats = autocontext_features(image_, labels_, train_cfg_ilastik)
    ilastik_feats = transfer_from_gpu(ilastik_feats)
    assert ilastik_feats.shape[0] == image_.shape[0] and ilastik_feats.shape[1] == image_.shape[1]


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


def test_train_and_apply_autocontext_modes(
    train_cfg: TrainingConfig,
) -> None:
    # Test end-to-end apply with autocontext='simple'
    cfg_simple = train_cfg.model_copy(update={"autocontext": "simple", "n_samples": 5000})
    feats = multiscale_features(image_, cfg_simple.feature_config)
    pred_simple, probs_simple, _ = train_and_apply(feats, labels_, cfg_simple, image=image_)
    assert pred_simple.shape == image_.shape
    assert probs_simple.shape == (image_.shape[0], image_.shape[1], 2)

    # Test end-to-end apply with autocontext='original'
    cfg_orig = train_cfg.model_copy(
        update={"autocontext": "original", "autocontext_distances": [5, 10], "autocontext_n_rays": 4, "n_samples": 5000}
    )
    pred_orig, probs_orig, _ = train_and_apply(feats, labels_, cfg_orig, image=image_)
    assert pred_orig.shape == image_.shape
    assert probs_orig.shape == (image_.shape[0], image_.shape[1], 2)


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


# ============================================================================
# SAM ONNX Tests
# ============================================================================


def test_sam_helpers() -> None:
    # 1. to_onnx_image with 2D array, 3D array, and PIL Image
    img_2d = (np.random.rand(64, 64) * 255).astype(np.uint8)
    img_3d = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_2d)

    for img in [img_2d, img_3d, img_pil]:
        onnx_img = to_onnx_image(img)
        assert onnx_img.shape == (1, 3, 64, 64)
        assert onnx_img.dtype == np.float32
        assert 0.0 <= onnx_img.min() and onnx_img.max() <= 1.0

    # 2. to_onnx_point_prompt & to_onnx_box_prompt formatting
    p_coords, p_labels = to_onnx_point_prompt([(10, 20)], [1])
    assert p_coords.shape == (1, 1, 1, 2) and p_labels.shape == (1, 1, 1)

    b_coords, b_labels = to_onnx_box_prompt([(10, 20, 30, 40)])
    assert b_coords.shape == (1, 1, 2, 2)
    assert np.array_equal(b_coords[0, 0], [[10, 20], [40, 60]])
    assert np.array_equal(b_labels[0, 0], [2.0, 3.0])

    # 3. Cache path and model loading error handling
    assert "models" in _get_default_cache_path_for_platform()
    with pytest.raises(FileNotFoundError):
        load_or_download_model("invalid_model_path.onnx", "encoder")


@pytest.mark.skipif(not SAM_AVAILABLE, reason="requires onnxruntime for SAM")
def test_sam_encoder_decoder(image: Arr) -> None:
    encoder = SAMEncoderONNX(None)
    decoder = SAMDecoderONNX(None)

    # 1. Check encoder embedding shape: (1, 256, 64, 64)
    embed = encoder.get_embedding(image)
    assert len(embed) == 1
    assert embed[0].shape == (1, 256, 64, 64)

    # 2. Point prompt (single-mask and multi-mask)
    pt = (200, 200)
    pt_mask, pt_score = decoder.masks_from_points(embed, image.shape, [pt], None, multimask_output=False)
    pt_masks, pt_scores = decoder.masks_from_points(embed, image.shape, [pt], None, multimask_output=True)
    assert pt_mask.shape == image.shape and pt_score > 0.0
    assert pt_masks.shape == (1, 1, 3, *image.shape) and pt_scores.shape == (3,)
    assert set(np.unique(pt_mask)).issubset({0.0, 1.0})

    # 3. Box prompt (single-mask and multi-mask)
    bbox = (80, 140, 200, 130)  # (x, y, w, h)
    box_mask, box_score = decoder.masks_from_boxes(embed, image.shape, [bbox], multimask_output=False)
    box_masks, box_scores = decoder.masks_from_boxes(embed, image.shape, [bbox], multimask_output=True)
    assert box_mask.shape == image.shape and box_score > 0.5
    assert box_masks.shape == (1, 1, 3, *image.shape) and box_scores.shape == (3,)
    assert set(np.unique(box_mask)).issubset({0.0, 1.0})


@pytest.mark.skipif(not SAM_AVAILABLE, reason="requires onnxruntime for SAM")
def test_do_sam_postproc(
    feature_stack: NPFloatArray,
    labels: NPUIntArray,
    image: Arr,
    train_cfg: TrainingConfig,
    ground_truth: NPUIntArray,
) -> None:
    # 1. Base model predictions
    seg, _, _ = train_and_apply(feature_stack, labels, train_cfg)

    # 2. End-to-end SAM post-processing
    class_infos = [ClassInfo(name="Secondary precipitate", value=1, do_sam_postproc=True, min_size_px=600)]
    postproc_seg = do_sam_postproc(seg, image, class_infos)
    assert postproc_seg.shape == seg.shape and postproc_seg.dtype == seg.dtype
    save_segmentation(postproc_seg, "tests/out/0_seg_sam.tif")

    # Assert mIoU improvement
    miou_base = class_avg_miou(seg, ground_truth)
    miou_sam = class_avg_miou(postproc_seg, ground_truth)
    assert miou_sam > miou_base

    # 3. Caching parity: passing precomputed embedding and regions
    encoder = SAMEncoderONNX(None)
    embed = encoder.get_embedding(image)
    cached_seg = do_sam_postproc(
        seg, image, class_infos, cached_embedding=embed, cached_regions=regionprops(label(seg))
    )
    assert np.array_equal(cached_seg, postproc_seg)

    # 4. Disabled flag & min size filter leave segmentation unchanged
    disabled_seg = do_sam_postproc(
        seg, image, [ClassInfo(name="Secondary precipitate", value=1, do_sam_postproc=False)]
    )
    assert np.array_equal(disabled_seg, seg)

    filtered_seg = do_sam_postproc(
        seg, image, [ClassInfo(name="Secondary precipitate", value=1, do_sam_postproc=True, min_size_px=1_000_000)]
    )
    assert np.array_equal(filtered_seg, seg)


if __name__ == "__main__":
    pytest.main(args=["-k test_extensions", "-s"])
