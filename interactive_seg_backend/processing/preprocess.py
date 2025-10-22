from typing import cast
from skimage.exposure import equalize_adapthist # type: ignore
from skimage.restoration import denoise_wavelet # type: ignore

from interactive_seg_backend.configs import Arr
from interactive_seg_backend.configs.types import (
    Preprocessing,
)
from interactive_seg_backend.utils import gaussian_ts


def preprocess(
    img_arr: Arr, preprocessing_operations: tuple[Preprocessing, ...]
) -> Arr:
    out: Arr
    if "blur" in preprocessing_operations:
        out = gaussian_ts(img_arr, 1)
    if "denoise" in preprocessing_operations:
        out = denoise_wavelet(img_arr, sigma=None, method="BayesShrink")
    if "equalize" in preprocessing_operations:
        out = equalize_adapthist(img_arr)
    out = cast(Arr, out)
    return out
