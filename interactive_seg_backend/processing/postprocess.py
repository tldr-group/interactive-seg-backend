from skimage.filters.rank import modal
from skimage.morphology import disk

from interactive_seg_backend.configs import UInt8Arr
from interactive_seg_backend.configs.types import (
    Postprocessing,
)

from typing import cast


def make_footprint(k: int) -> UInt8Arr:
    out = disk(2 * k + 1)
    return cast(UInt8Arr, out)


def modal_filter(seg_arr: UInt8Arr, k: int = 2) -> UInt8Arr:
    footprint = make_footprint(k)
    out = modal(seg_arr, footprint)
    return cast(UInt8Arr, out)


def postprocess(seg_arr: UInt8Arr, postprocessing_operations: tuple[Postprocessing]) -> UInt8Arr:
    if "modal_filter" in postprocessing_operations:
        seg_arr = modal_filter(seg_arr)
    return seg_arr
