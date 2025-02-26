from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_wavelet
from skimage.filters import gaussian

from interactive_seg_backend.configs import Arr
from interactive_seg_backend.configs.types import (
    Preprocessing,
)


def preprocess(img_arr: Arr, preprocessing_operations: tuple[Preprocessing]) -> Arr:
    if "blur" in preprocessing_operations:
        img_arr = gaussian(img_arr, 1)
    if "denoise" in preprocessing_operations:
        img_arr = denoise_wavelet(img_arr, sigma=0.12, mode="BayesShrink")
    if "equalize" in preprocessing_operations:
        img_arr = equalize_adapthist(img_arr)
    return img_arr
