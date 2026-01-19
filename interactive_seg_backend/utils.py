from PIL.ImageChops import add
import numpy as np
from scipy.ndimage import rotate
from skimage.filters import gaussian

import logging
from sys import stdout
from time import strftime, localtime


from interactive_seg_backend.configs import NPFloatArray, NPIntArray


def class_avg_mious(prediction: NPIntArray, ground_truth: NPIntArray) -> list[float]:
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


def class_avg_miou(prediction: NPIntArray, ground_truth: NPIntArray) -> float:
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
    input: NPFloatArray,
    angle: float,
    axes: tuple[int, ...] = (1, 0),
    reshape: bool = True,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0,
    prefilter: bool = True,
) -> NPFloatArray:
    return rotate(input, angle, axes, reshape, None, order, mode, cval, prefilter)


def gaussian_ts(
    image: NPFloatArray,
    sigma: float = 1,
    mode: str = "nearest",
    cval: int = 0,
    preserve_range: bool = False,
    truncate: float = 4,
    channel_axis: int | None = None,
) -> NPFloatArray:
    return gaussian(
        image,
        sigma,
        mode=mode,
        cval=cval,
        preserve_range=preserve_range,
        truncate=truncate,
        channel_axis=channel_axis,
    )


def add_color(string: str, color_code: str) -> str:
    RESET_CODE = "\033[0m"
    return f"{color_code}{string}{RESET_CODE}"


class ColorFormatter(logging.Formatter):
    TIME_COLOR = "\033[90m"  # gray
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"
    CHARS = {
        logging.DEBUG: "D",
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.CRITICAL: "C",
    }

    def format(self, record: logging.LogRecord):
        fmt = self.datefmt if self.datefmt else self.default_time_format
        timestamp = strftime(fmt, localtime(record.created))

        color = self.COLORS.get(record.levelno, "")
        record.filename = add_color(record.filename, color)
        record.levelname = add_color(self.CHARS[record.levelno], color)

        record.asctime_coloured = add_color(timestamp, self.TIME_COLOR)
        record.lineno_coloured = add_color(str(record.lineno), self.TIME_COLOR)

        return super().format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(stdout)

formatter = ColorFormatter(
    fmt="%(asctime_coloured)s | %(levelname)s | %(filename)s:%(lineno_coloured)s -  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

handler.setFormatter(formatter)
logger.addHandler(handler)


if __name__ == "__main__":
    test = np.zeros((100, 100))
    test = to_rgb_arr(test)
    print(test.shape)

    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
