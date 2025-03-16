import numpy as np
import numpy.typing as npt
from PIL import Image
from tifffile import imread, imwrite, COMPRESSION
from os.path import exists

from typing import cast, Literal

from interactive_seg_backend.configs import Arr, Arrlike


def read_file_get_arr(path: str) -> Arr:
    if not exists(path):
        raise FileNotFoundError

    file_ext = path.split(".")[-1].lower()
    if file_ext in ("png", "jpg", "jpeg"):
        img = Image.open(path)
        arr = np.array(img)
    elif file_ext in ("tif", "tiff"):
        arr = imread(path)
    else:
        raise Exception(f"filetype '.{file_ext}' not supported!")

    return arr


def load_featurestack(path: str) -> Arrlike:
    file_ext = path.split(".")[-1].lower()
    stack: Arrlike
    if file_ext in ("tif", "tiff"):
        _stack = imread(path)
        stack = cast(Arrlike, _stack)
    elif file_ext in ("npy", "npz"):
        stack = np.load(path)
    else:
        raise Exception(f"filetype '.{file_ext}' not supported!")
    return stack


def save_featurestack(
    arr: Arrlike, path: str, save_types: Literal[".npy", ".npz", ".tif", ".pt"]
) -> None:
    file_ext = path.split(".")[-1].lower()
    if save_types == ".npy":
        np.save(path, arr)
    elif save_types == ".npz":
        np.savez_compressed(path, arr)
    elif save_types == ".tif":
        imwrite(path, arr)
    else:
        raise Exception(f"filetype '.{file_ext}' not supported!")


def rescale_labels_to_greyscale(
    labels: npt.NDArray[np.uint8], offset: int = 1, n_classes: int | None = None
) -> npt.NDArray[np.uint8]:
    # offset useful if seg 0 indexed. if remapping labels, set this to 0
    if n_classes is None:
        amax = int(np.max(labels.flatten()))
        n_classes = amax
    delta = 255 // (n_classes + offset)
    rescaled = (labels + offset) * delta
    return rescaled.astype(np.uint8)


N_VALS_CUTOFF = 20  # if they have more than 20 classes in arr, throw eror


def rescale_unique_vals_to_contiguous_labels(
    arr: Arr,
) -> npt.NDArray[np.uint8]:
    # map from (2D) np array, go from unique values -> classes
    unique_vals = sorted(np.unique(arr))
    if len(unique_vals) > N_VALS_CUTOFF:
        raise Exception(
            "Too many unique values in array! Are you sure this is an integer label array? "
        )

    out = np.zeros_like(arr, dtype=np.uint8)
    for i, val in enumerate(unique_vals):
        out = np.where(arr == val, i, out)
    return out


def rescale_RGB_to_contiguous_labels(
    arr: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    # convert RGB arrs -> unique ints -> contiguous labels
    new_arr = arr.astype(np.int64)
    R, G, B = new_arr[:, :, 0], new_arr[:, :, 1], new_arr[:, :, 2]
    unique_mapping = R + 255 * G + (255**2) * B
    return rescale_unique_vals_to_contiguous_labels(unique_mapping)


def load_labels(path: str) -> npt.NDArray[np.uint8]:
    arr = read_file_get_arr(path).astype(np.uint8)
    is_RGB = len(arr.shape) == 3 and arr.shape[-1] == 3
    if is_RGB:
        return rescale_RGB_to_contiguous_labels(arr)
    else:
        return rescale_unique_vals_to_contiguous_labels(arr)


def load_image(path: str) -> Arr:
    return read_file_get_arr(path)


def save_segmentation(
    arr: npt.NDArray[np.uint8],
    out_path: str,
    rescale: bool = True,
    as_png: bool = False,
    offset: int = 1,
) -> None:
    to_save: npt.NDArray[np.uint8]
    if rescale:
        to_save = rescale_labels_to_greyscale(arr, offset=offset)
    else:
        to_save = arr

    if as_png:
        img = Image.fromarray(to_save)
        img.save(out_path)
    else:
        imwrite(out_path, to_save, compression=COMPRESSION.DEFLATE)


def save_labels(
    arr: npt.NDArray[np.uint8],
    out_path: str,
    rescale: bool = True,
    as_png: bool = False,
) -> None:
    save_segmentation(arr, out_path, rescale, as_png, offset=0)
