from skimage.filters.rank import modal
from skimage.morphology import disk

from interactive_seg_backend.configs import NPUIntArray
from interactive_seg_backend.configs.types import (
    Postprocessing,
)

from typing import cast


def make_footprint(k: int) -> NPUIntArray:
    out = disk(2 * k + 1)
    return cast(NPUIntArray, out)


def modal_filter(seg_arr: NPUIntArray, k: int = 2) -> NPUIntArray:
    footprint = make_footprint(k)
    out = modal(seg_arr, footprint)
    return cast(NPUIntArray, out)


def postprocess(seg_arr: NPUIntArray, postprocessing_operations: tuple[Postprocessing]) -> NPUIntArray:
    if "modal_filter" in postprocessing_operations:
        seg_arr = modal_filter(seg_arr)
    return seg_arr
