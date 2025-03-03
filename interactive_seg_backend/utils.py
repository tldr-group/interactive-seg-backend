import numpy as np

from interactive_seg_backend.configs import UInt8Arr


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


if __name__ == "__main__":
    test = np.zeros((100, 100))
    test = to_rgb_arr(test)
    print(test.shape)
