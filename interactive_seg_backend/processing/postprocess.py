from skimage.filters.rank import modal  # type: ignore[import]
from skimage.morphology import disk  # type: ignore[import]

from interactive_seg_backend.configs import UInt8Arr
from interactive_seg_backend.configs.types import (
    Postprocessing,
)


def make_footprint(k: int) -> UInt8Arr:
    return disk(2 * k + 1)  # type: ignore


def modal_filter(seg_arr: UInt8Arr, k: int = 2) -> UInt8Arr:
    footprint = make_footprint(k)
    return modal(seg_arr, footprint)  # type: ignore


def postprocess(
    seg_arr: UInt8Arr, postprocessing_operations: tuple[Postprocessing]
) -> UInt8Arr:
    if "modal_filter" in postprocessing_operations:
        seg_arr = modal_filter(seg_arr)
    return seg_arr
