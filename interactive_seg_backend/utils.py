import numpy as np

from interactive_seg_backend.configs import UInt8Arr


def class_avg_mious(prediction: UInt8Arr, ground_truth: UInt8Arr) -> list[float]:
    ious: list[float] = []
    vals = np.unique(ground_truth)
    for v in vals:
        mask_pred = np.where(prediction == v, 1, 0)
        mask_gt = np.where(ground_truth == v, 1, 0)
        overlap = mask_pred * mask_gt
        union = mask_pred & mask_gt
        iou = float(np.sum(overlap) / np.sum(union))
        ious.append(iou)
    return ious


def class_avg_miou(prediction: UInt8Arr, ground_truth: UInt8Arr) -> float:
    mious = class_avg_mious(prediction, ground_truth)
    mean = np.mean(mious)
    return float(mean)
