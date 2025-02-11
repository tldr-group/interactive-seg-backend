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
from skimage.util.dtype import img_as_float32

from itertools import combinations_with_replacement
from itertools import chain

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from interactive_seg_backend.configs import FeatureConfig

from typing import Literal


# - 2 to allow for main & gui threads
BACKEND: Literal["loky", "threading"] = "threading"
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
    img: npt.NDArray[np.float32], sigma: int, mult: float = 1.0
) -> npt.NDArray[np.float32]:
    """Gaussian blur of each pixel in $img of scale/radius $sigma.

    :param img: img arr
    :type img: np.ndarray
    :param sigma: radius for footprint
    :type sigma: int
    :return: filtered array
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.float32] = filters.gaussian(
        img, sigma=int(mult * sigma), preserve_range=True, truncate=2.0
    )
    return out


def singlescale_edges(
    gaussian_filtered: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Sobel filter applied to gaussian filtered arr of scale sigma to detect edges.

    :param gaussian_filtered: img array (that has optionally been gaussian blurred)
    :type gaussian_filtered: np.ndarray
    :return: sobel filtered (edge-detecting) array
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.float32] = filters.sobel(gaussian_filtered)
    return out


def singlescale_hessian(
    gaussian_filtered: npt.NDArray[np.float32], return_full: bool = True
) -> tuple[npt.NDArray[np.float32], ...]:
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
        return (eig1 / 2.0, eig2 / 2.0, mod, trace, det)
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
    img: npt.NDArray[np.float32], sigma: int
) -> tuple[npt.NDArray[np.float32], ...]:
    """Compute structure tensor eigenvalues of $img in $sigma radius.

    :param img: img arr
    :type img: np.ndarray
    :param sigma: scale parameter
    :type sigma: int
    :return: largest two eigenvalues of structure tensor at each pixel
    :rtype: np.ndarray
    """
    tensor: list[npt.NDArray[np.float32]] = structure_tensor(img, sigma)
    eigvals: npt.NDArray[np.float32] = structure_tensor_eigenvalues(tensor)
    return (eigvals[0], eigvals[1])


def singlescale_laplacian(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
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
def bilateral(byte_img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """For $sigma in [5, 10], for $value_range in [50, 100],
        compute mean of pixels in $sigma radius inside $value_range window for each pixel.

    :param img: img arr
    :type img: np.ndarray
    :return: bilateral filtered arrs stacked in a single np array
    :rtype: np.ndarray
    """
    bilaterals: list[npt.NDArray[np.uint8]] = []
    for spatial_radius in (5, 10):
        footprint = make_footprint(spatial_radius)
        for value_range in (50, 100):  # check your pixels are [0, 255]
            filtered: npt.NDArray[np.uint8] = filters.rank.mean_bilateral(
                byte_img, footprint, s0=value_range, s1=value_range
            )
            bilaterals.append(filtered)
    return np.stack(bilaterals, axis=0)


def difference_of_gaussians(
    gaussian_blurs: list[npt.NDArray[np.float32]],
) -> list[npt.NDArray[np.float32]]:
    """Compute their difference of each arr in $gaussian_blurs (representing different $sigma scales) with smaller arrs.

    :param gaussian_blurs: list of arrs of img filtered with gaussian blur at different length scales
    :type gaussian_blurs: List[np.ndarray]
    :return: list of differences of each blurred img with smaller length scales.
    :rtype: List[np.ndarray]
    """
    # weka computes dog for  each filter of a *lower* sigma
    dogs: list[npt.NDArray[np.float32]] = []
    for i in range(len(gaussian_blurs)):
        sigma_1 = gaussian_blurs[i]
        for j in range(i):
            sigma_2 = gaussian_blurs[j]
            dogs.append(sigma_2 - sigma_1)
    return dogs


def membrane_projections(
    img: npt.NDArray[np.float32],
    membrane_patch_size: int = 19,
    membrane_thickness: int = 1,
    num_workers: int | None = N_ALLOWED_CPUS,
) -> list[npt.NDArray[np.float32]]:
    """Membrane projections.

    Create a $membrane_patch_size^2 array with $membrane_thickness central columns set to 1, other entries set to 0.
    Next compute 30 different rotations of membrane kernel ($theta in [0, 180, step=6 degrees]).
    Convolve each of these kernels with $img to get HxWx30 array, then z-project the array by taking
    the sum, mean, std, median, max and min to get a HxWx6 array out.

    :param img: img arr
    :type img: np.ndarray
    :param membrane_patch_size: size of kernel, defaults to 19
    :type membrane_patch_size: int, optional
    :param membrane_thickness: width of line down the middle, defaults to 1
    :type membrane_thickness: int, optional
    :param num_workers: number of threads, defaults to N_ALLOWED_CPUS
    :type num_workers: int | None, optional
    :return: List of 6 z-projections of membrane convolutions
    :rtype: List[np.ndarray]
    """
    kernel = np.zeros((membrane_patch_size, membrane_patch_size))
    x0 = membrane_patch_size // 2 - membrane_thickness // 2
    x1 = 1 + membrane_patch_size // 2 + membrane_thickness // 2
    kernel[:, x0:x1] = 1

    all_kernels = [np.rint(rotate(kernel, angle)) for angle in range(0, 180, 6)]
    # map these across threads to speed up (order unimportant)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        out_angles: list[npt.NDArray[np.float32]] = list(
            ex.map(
                lambda k: convolve(img, k),
                all_kernels,
            )
        )
    out_angles_np = np.stack(out_angles, axis=0)
    sum_proj = np.sum(out_angles_np, axis=0)
    mean_proj = np.mean(out_angles_np, axis=0)
    std_proj = np.std(out_angles_np, axis=0)
    median_proj = np.median(out_angles_np, axis=0)
    max_proj = np.amax(out_angles_np, axis=0)
    min_proj = np.amin(out_angles_np, axis=0)
    return [mean_proj, max_proj, min_proj, sum_proj, std_proj, median_proj]


# # %% ===================================MANAGER FUNCTIONS===================================


def singlescale_singlechannel_features(
    raw_img: npt.NDArray[np.uint8], sigma: int, config: FeatureConfig
):
    assert len(raw_img.shape) == 2, (
        f"img shape {raw_img.shape} wrong, should be 2D/singlechannel"
    )
    img: npt.NDArray[np.float32] = np.ascontiguousarray(img_as_float32(raw_img))
    results: list[npt.NDArray[np.uint8 | np.float32]] = []
    gaussian_filtered = singlescale_gaussian(img, sigma)
    if config.gaussian_blur:
        results.append(gaussian_filtered)
    if config.sobel_filter:
        results.append(singlescale_edges(gaussian_filtered))
    if config.hessian_filter:
        hessian_out = singlescale_hessian(
            gaussian_filtered, config.add_mod_trace_det_hessian
        )
        results += hessian_out

    byte_img = raw_img.astype(np.uint8)
    circle_footprint = make_footprint(int(np.ceil(sigma)))

    if config.mean:
        results.append(singlescale_mean(byte_img, circle_footprint))
    if config.median:
        results.append(singlescale_median(byte_img, circle_footprint))
    if config.maximum:
        results.append(singlescale_maximum(byte_img, circle_footprint))
    if config.minimum:
        results.append(singlescale_minimum(byte_img, circle_footprint))

    if config.laplacian:
        results.append(singlescale_laplacian(gaussian_filtered))
    if config.structure_tensor_eigvals:
        structure_out = singlescale_structure_tensor(img, sigma)
        results += structure_out

    return results


def zero_scale_filters(
    img: npt.NDArray[np.float32],
    sobel_filter: bool = True,
    hessian_filter: bool = True,
    add_mod_trace: bool = True,
) -> list[npt.NDArray[np.float32]]:
    """Weka *always* adds the original image, and if computing edgees and/or hessian,
    adds those for sigma=0. This function does that."""
    out_filtered: list[npt.NDArray[np.float32]] = [img]
    if sobel_filter:
        edges = singlescale_edges(img)
        out_filtered.append(edges)
    if hessian_filter:
        hessian = singlescale_hessian(img, add_mod_trace)
        out_filtered += hessian
    return out_filtered


def multiscale_features(
    raw_img: npt.NDArray[np.uint8],
    config: FeatureConfig,
    num_workers: int | None = None,
) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
    converted_img: npt.NDArray[np.float32] = np.ascontiguousarray(
        img_as_float32(raw_img)  # type: ignore
    )
    features: list[npt.NDArray[np.float32]]
    if config.add_zero_scale_features:
        features = zero_scale_filters(
            converted_img,
            config.sobel_filter,
            config.hessian_filter,
            config.add_mod_trace_det_hessian,
        )
    else:
        features = []

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        out_sigmas = list(
            ex.map(
                lambda sigma: singlescale_singlechannel_features(
                    raw_img, sigma, config
                ),
                config.sigmas,
            )
        )

    multiscale_features = chain.from_iterable(out_sigmas)
    features += list(multiscale_features)  # type: ignore

    if config.difference_of_gaussians:
        intensities: list[npt.NDArray[np.float32]] = []
        for i in range(len(config.sigmas)):
            gaussian_blur_at_sigma: npt.NDArray[np.float32] = out_sigmas[i][0]  # type: ignore
            intensities.append(gaussian_blur_at_sigma)
        dogs = difference_of_gaussians(intensities)
        features += dogs

    if config.membrane_projections:
        projections = membrane_projections(
            converted_img,
            config.membrane_patch_size,
            config.membrane_thickness,
            num_workers,
        )
        features += projections

    if config.bilateral:
        byte_img = img.astype(np.uint8)
        bilateral_filtered = bilateral(byte_img)
        features += bilateral_filtered

    features_np: npt.NDArray[np.float16 | np.float32 | np.float64] = np.stack(
        features, axis=-1
    )
    if config.cast_to == "f16":
        features_np = features_np.astype(np.float16)
    elif config.cast_to == "f64":
        features_np = features_np.astype(np.float64)
    else:
        features_np = features_np.astype(np.float32)
    return features_np


if __name__ == "__main__":
    cfg = FeatureConfig(laplacian=True, structure_tensor_eigvals=True, cast_to="f16")
    img = (np.random.uniform(0, 1.0, (1000, 1000)) * 255).astype(np.uint8)
    feats = multiscale_features(img, cfg, num_workers=N_ALLOWED_CPUS)
    print(feats.shape)
