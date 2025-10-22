import numpy as np
from scipy.ndimage import rotate  # type: ignore[import]
from skimage.filters import gaussian  # type: ignore[import]

from interactive_seg_backend.configs.types import Arr, FloatArr, UInt8Arr


def class_avg_mious(prediction: UInt8Arr, ground_truth: UInt8Arr) -> list[float]:
    ious: list[float] = []
    vals = np.unique(ground_truth)
    for v in vals:
        mask_pred = np.where(prediction == v, 1, 0)
        mask_gt = np.where(ground_truth == v, 1, 0)
        overlap = np.logical_and(mask_pred, mask_gt)
        union = np.logical_or(mask_pred, mask_gt)
        iou = float(np.sum(overlap) / np.sum(union))
        ious.append(iou)
    return ious


def class_avg_miou(prediction: UInt8Arr, ground_truth: UInt8Arr) -> float:
    mious = class_avg_mious(prediction, ground_truth)
    mean = np.mean(mious)
    return float(mean)


def to_rgb_arr(arr: np.ndarray) -> np.ndarray:
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, -1)
    elif len(arr.shape) == 3 and arr.shape[0] == 1:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.repeat(arr, 3, axis=-1)
    return arr


# ========== TYPESAFE WRAPPERS ==========
# wrappers to make the typechecker happy


def rotate_ts(
    input: FloatArr,
    angle: float,
    axes: tuple[int, ...] = (1, 0),
    reshape: bool = True,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0,
    prefilter: bool = True,
) -> FloatArr:
    return rotate(input, angle, axes, reshape, None, order, mode, cval, prefilter)  # type: ignore


def gaussian_ts(
    image: Arr,
    sigma: float = 1,
    mode: str = "nearest",
    cval: int = 0,
    preserve_range: bool = False,
    truncate: float = 4,
    channel_axis: int | None = None,
) -> FloatArr:
    return gaussian(
        image,
        sigma,
        mode=mode,
        cval=cval,
        preserve_range=preserve_range,
        truncate=truncate,
        channel_axis=channel_axis,
    )  # type: ignore


if __name__ == "__main__":
    test = np.zeros((100, 100))
    test = to_rgb_arr(test)
    print(test.shape)
