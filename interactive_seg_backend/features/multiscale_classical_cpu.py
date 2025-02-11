"""
2D multi-scale featurisation of a single channel image.

Approach inspired by (1)
https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.multiscale_basic_features
Designed to be a Python equivalent of (most) of the features present at (2) https://imagej.net/plugins/tws/
Heavy use of skimage filters, filters.rank and feature.
General approach is:
• for each $sigma (a scale over which to compute a feature for a pixel):
    • compute each singlescale singlechannel feature
• compute scale free features (difference of Gaussians, Membrane Projections, Bilateral)
• combine, stack as np array in form (HxWxN_features)

Singlescale feature computation is mapped over multiple threads as in (1).
Every feature computes a value for *every pixel* in the image.
"""

import numpy as np
import numpy.typing as npt

from skimage import filters
from skimage.draw import disk
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from scipy.ndimage import rotate, convolve  # type: ignore


from itertools import combinations_with_replacement
# from skimage import filters, feature
# from skimage.util.dtype import img_as_float32

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from typing import Literal


# - 2 to allow for main & gui threads
BACKEND: Literal["loky", "threading"] = "loky"
N_ALLOWED_CPUS = cpu_count() - 2

print(f"N CPUS: {N_ALLOWED_CPUS}")


# %% ===================================HELPER FUNCTIONS===================================
def make_footprint(sigma: int) -> npt.NDArray[np.uint8]:
    """Return array of zeros with centreed circle of radius sigma set to 1.

    :param sigma: radius of footprint
    :type sigma: int
    :return: array with disk radius sigma set to 1
    :rtype: np.ndarray
    """
    circle_footprint = np.zeros((2 * sigma + 1, 2 * sigma + 1), dtype=np.uint8)
    centre = (sigma, sigma)
    rr: npt.NDArray[np.int64]
    cc: npt.NDArray[np.int64]
    rr, cc = disk(centre, sigma)
    circle_footprint[rr, cc] = 1
    return circle_footprint


# %% ===================================SINGLESCALE FEATURES===================================


def singlescale_gaussian(
    img: npt.NDArray[np.float64], sigma: int, mult: float = 1.0
) -> npt.NDArray[np.float64]:
    """Gaussian blur of each pixel in $img of scale/radius $sigma.

    :param img: img arr
    :type img: np.ndarray
    :param sigma: radius for footprint
    :type sigma: int
    :return: filtered array
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.float64] = filters.gaussian(
        img, sigma=int(mult * sigma), preserve_range=True, truncate=2.0
    )
    return out


def singlescale_edges(
    gaussian_filtered: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Sobel filter applied to gaussian filtered arr of scale sigma to detect edges.

    :param gaussian_filtered: img array (that has optionally been gaussian blurred)
    :type gaussian_filtered: np.ndarray
    :return: sobel filtered (edge-detecting) array
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.float64] = filters.sobel(gaussian_filtered)
    return out


def singlescale_hessian(
    gaussian_filtered: npt.NDArray[np.float64], return_full: bool = True
) -> tuple[npt.NDArray[np.float64], ...]:
    """Compute mod, trace, det and eigenvalues of Hessian matrix of $gaussian_filtered image (i.e for every pixel).

    :param gaussian_filtered: img array (that has optionally been gaussian blurred)
    :type gaussian_filtered: np.ndarray
    :return: 5 arrays the same shape as input that are the module, trace, determinant and first 2 eigenvalues
        of the hessian at that pixel
    :rtype: Tuple[np.ndarray, ...]
    """
    H_elems = [
        np.gradient(np.gradient(gaussian_filtered)[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(range(gaussian_filtered.ndim), 2)
    ]
    a, b, d = H_elems
    mod = np.sqrt(a**2 + b**2 + d**2)
    trace = a + d
    det = a * d - b**2
    # orientation_2 = orientation_1 + np.pi / 2
    # eigvals = feature.hessian_matrix_eigvals(H_elems)
    eig1 = trace + np.sqrt((4 * b**2 + (a - d) ** 2))
    eig2 = trace - np.sqrt((4 * b**2 + (a - d) ** 2))

    if return_full:
        return (mod, trace, det, eig1 / 2.0, eig2 / 2.0)
    else:
        return (eig1 / 2.0, eig2 / 2.0)


def singlescale_mean(
    byte_img: npt.NDArray[np.uint8], sigma_rad_footprint: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Mean pixel intensity over footprint $sigma_rad_footprint. Needs img in np.uint8 format.

    :param byte_img: img arr in uint8 format
    :type byte_img: np.ndarray
    :param sigma_rad_footprint: radius of footprint
    :type sigma_rad_footprint: np.ndarray
    :return: mean filtered img
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.uint8] = filters.rank.mean(byte_img, sigma_rad_footprint)
    return out


def singlescale_median(
    byte_img: npt.NDArray[np.uint8], sigma_rad_footprint: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Median pixel intensity over footprint $sigma_rad_footprint. Needs img in np.uint8 format.

    :param byte_img: img arr in uint8 format
    :type byte_img: np.ndarray
    :param sigma_rad_footprint: radius of footprint
    :type sigma_rad_footprint: np.ndarray
    :return: mean filtered img
    :rtype: np.ndarray
    """
    return filters.rank.median(byte_img, sigma_rad_footprint)


def singlescale_maximum(
    byte_img: npt.NDArray[np.uint8], sigma_rad_footprint: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """maximum pixel intensity over footprint $sigma_rad_footprint. Needs img in np.uint8 format.

    :param byte_img: img arr in uint8 format
    :type byte_img: np.ndarray
    :param sigma_rad_footprint: radius of footprint
    :type sigma_rad_footprint: np.ndarray
    :return: maximum filtered img
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.uint8] = filters.rank.maximum(byte_img, sigma_rad_footprint)
    return out


def singlescale_minimum(
    byte_img: npt.NDArray[np.uint8], sigma_rad_footprint: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """maximum pixel intensity over footprint $sigma_rad_footprint. Needs img in np.uint8 format.

    :param byte_img: img arr in uint8 format
    :type byte_img: np.ndarray
    :param sigma_rad_footprint: radius of footprint
    :type sigma_rad_footprint: np.ndarray
    :return: minimum filtered img
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.uint8] = filters.rank.minimum(byte_img, sigma_rad_footprint)
    return out


def singlescale_structure_tensor(
    img: npt.NDArray[np.float64], sigma: int
) -> npt.NDArray[np.float64]:
    """Compute structure tensor eigenvalues of $img in $sigma radius.

    :param img: img arr
    :type img: np.ndarray
    :param sigma: scale parameter
    :type sigma: int
    :return: largest two eigenvalues of structure tensor at each pixel
    :rtype: np.ndarray
    """
    tensor: list[npt.NDArray[np.float64]] = structure_tensor(img, sigma)
    eigvals: npt.NDArray[np.float64] = structure_tensor_eigenvalues(tensor)
    return eigvals[:2]


def singlescale_laplacian(img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute laplacian of $img on scale $simga. Not currently working.

    :param img: img arr
    :type img: np.ndarray
    :param sigma: scale parameter
    :type sigma: int
    :return: laplacian filtered img arr
    :rtype: np.ndarray
    """
    return filters.laplace(img)


# # %% ===================================SCALE-FREE FEATURES===================================
