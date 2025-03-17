import numpy as np
from scipy.ndimage import rotate  # type: ignore
import torch
from torch.nn.functional import conv2d, max_pool2d, avg_pool2d, pad
from kornia.filters import (
    gaussian_blur2d,
    median_blur,
    laplacian,
    bilateral_blur,
    get_gaussian_kernel2d,
)

from time import time

from typing import cast

from interactive_seg_backend.configs import FeatureConfig, Arr, Arrlike


def prepare_for_gpu(
    arr: Arr, device: str = "cuda:0", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    ndims = len(arr.shape)
    arr = cast(np.ndarray, arr)
    if ndims == 2:
        arr = np.expand_dims(arr, (0, 1))  # (H, W) -> (1, 1, H, W)
    else:
        channel_idx = np.argmin(arr.shape)
        if channel_idx == ndims - 1:  # (H, W, C) -> (C, H, W)
            arr = np.transpose(arr, (-1, 0, 1))
        arr = np.expand_dims(arr, (0))  # (C, H, W) -> (1, 1, H, W)
    tensor = torch.tensor(arr, device=device, dtype=dtype)
    return tensor


def concat_feats(arr1: Arrlike, arr2: Arrlike) -> Arrlike:
    # (optionally) cast to tensors and concatenate arrays
    if isinstance(arr1, torch.Tensor) and isinstance(arr2, torch.Tensor):
        res = torch.concatenate((arr1, arr2), dim=-1)
    elif isinstance(arr1, torch.Tensor) and not isinstance(arr2, torch.Tensor):
        tensor_2 = torch.tensor(arr2, dtype=arr1.dtype, device=arr1.device)
        res = torch.concatenate((arr1, tensor_2), dim=-1)
    elif isinstance(arr2, torch.Tensor) and not isinstance(arr1, torch.Tensor):
        tensor_1 = torch.tensor(arr1, dtype=arr2.dtype, device=arr2.device)
        res = torch.concatenate((tensor_1, arr2), dim=-1)
    elif isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        res = np.concatenate((arr1, arr2), axis=-1)
    else:
        raise Exception(f"Invalid feat types: {type(arr1)} + {type(arr2)}")

    return cast(Arrlike, res)


# %% ===================================SINGLESCALE FEATURES===================================
def singlescale_gaussian(
    img: torch.Tensor, sigma: int, mult: float = 1.0
) -> torch.Tensor:
    s = int(mult * sigma)
    out = gaussian_blur2d(img, kernel_size=(2 * s + 1, 2 * s + 1), sigma=(s, s))
    return out


def get_multiscale_gaussian_kernel(
    device: torch.device,
    dtype: torch.dtype,
    sigmas: tuple[float, ...],
    n_channels: int,
    mult: float = 1.0,
) -> torch.Tensor:
    # get kernel of shape (N_s, max_k, max_k) where max_k is largest (truncated) gaussian kernel
    N = len(sigmas)
    max_s = max(sigmas)
    max_k = 4 * int(max_s * mult) + 1
    filters = torch.zeros(
        (N, 1, max_k, max_k), dtype=dtype, device=device, requires_grad=False
    )
    for i, sigma in enumerate(sigmas):
        filters[i, :, :, :] = get_gaussian_kernel2d(
            (max_k, max_k), (sigma * mult, sigma * mult), device=device, dtype=dtype
        )
    filters = torch.tile(filters, (n_channels, 1, 1, 1))
    return filters


def get_sobel_kernel(
    device: torch.device, dtype: torch.dtype, n_channels: int
) -> torch.Tensor:
    g_y = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    g_x = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    stacked = torch.stack((g_x, g_y))
    filters = stacked.unsqueeze(1)
    filters = torch.tile(filters, (n_channels, 1, 1, 1))
    return filters


def reflect_padded(func):  # type: ignore
    def wrapper(
        tensor: torch.Tensor,
        sigma_or_kernel: int | torch.Tensor,
        *args,  # type: ignore
        **kwargs,  # type: ignore
    ) -> torch.Tensor:
        if isinstance(sigma_or_kernel, torch.Tensor):
            _, _, kh, kw = sigma_or_kernel.shape
        else:
            kh, kw = 2 * sigma_or_kernel + 1, 2 * sigma_or_kernel + 1
        padded = pad(tensor, (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
        return func(padded, sigma_or_kernel, *args, **kwargs)

    return wrapper


@reflect_padded
def convolve(
    img: torch.Tensor, kernel: torch.Tensor, norm: bool = False
) -> torch.Tensor:
    _, in_ch, _, _ = img.shape
    if norm:
        summand = torch.sum(torch.abs(kernel), dim=(2, 3), keepdim=True)
        kernel_norm = kernel / summand
    else:
        kernel_norm = kernel

    convolved = conv2d(img, kernel_norm, stride=1, groups=in_ch)
    return convolved


def get_gradient_mag(edges: torch.Tensor) -> torch.Tensor:
    g_x = edges[0:1, 0::2]
    g_y = edges[0:1, 1::2]
    return torch.sqrt((g_x**2 + g_y**2))


def singescale_hessian(
    dx_dy: torch.Tensor, sobel_kernel: torch.Tensor, return_full: bool = True
) -> torch.Tensor:
    """_summary_

    :param dx_dy: (B, 2, H, W) first derivatives from sobel
    :type dx_dy: torch.Tensor
    :param sobel_kernel: (2, 1, 3, 3) sobel kernel
    :type sobel_kernel: torch.Tensor
    :param return_full: return mod, det and trace as well as eigs, defaults to True
    :type return_full: bool, optional
    :return: either
    :rtype: torch.Tensor
    """
    second_deriv = convolve(dx_dy, sobel_kernel, True)

    a: torch.Tensor = second_deriv[0:1, 0::4]
    b: torch.Tensor = second_deriv[0:1, 1::4]
    d: torch.Tensor = second_deriv[0:1, 3::4]

    mod = torch.sqrt(a**2 + b**2 + d**2)
    trace = a + d
    det = a * d - b**2

    eig1 = trace + torch.sqrt((4 * b**2 + (a - d) ** 2))
    eig2 = trace - torch.sqrt((4 * b**2 + (a - d) ** 2))

    to_stack: tuple[torch.Tensor, ...]
    if return_full:
        to_stack = (eig1 / 2.0, eig2 / 2.0, mod, trace, det)
    else:
        to_stack = (eig1 / 2.0, eig2 / 2.0)
    out = torch.cat(to_stack, dim=1)
    return out


@reflect_padded
def singlescale_mean(img: torch.Tensor, sigma: int) -> torch.Tensor:
    k = 2 * sigma + 1
    out = avg_pool2d(img, k, 1, ceil_mode=True)
    return out


@reflect_padded
def singlescale_maximum(img: torch.Tensor, sigma: int) -> torch.Tensor:
    k = 2 * sigma + 1
    out = max_pool2d(img, k, 1, ceil_mode=True)
    return out


@reflect_padded
def singlescale_minimum(img: torch.Tensor, sigma: int) -> torch.Tensor:
    k = 2 * sigma + 1
    out = -max_pool2d(-img, k, 1, ceil_mode=True)
    return out


def singlescale_median(img: torch.Tensor, sigma: int) -> torch.Tensor:
    k = 2 * sigma + 1
    return median_blur(img, k)


def singlescale_laplacian(img: torch.Tensor, sigma: int) -> torch.Tensor:
    k = 2 * sigma + 1
    return laplacian(img, (k, k))


# %% ===================================SCALE-FREE FEATURES===================================
def bilateral(img: torch.Tensor) -> torch.Tensor:
    bilaterals: list[torch.Tensor] = []
    for spatial_radius in (3, 5):
        for value_range in (50, 100):  # check your pixels are [0, 255]
            k = 2 * spatial_radius + 1
            filtered: torch.Tensor = bilateral_blur(
                img,
                k,
                sigma_color=value_range / 255.0,
                sigma_space=(spatial_radius, spatial_radius),
            )
            bilaterals.append(filtered)
    return torch.cat(bilaterals, dim=1)


def difference_of_gaussians(
    gaussian_blurs: torch.Tensor, N_sigmas: int
) -> torch.Tensor:
    diff_list: list[torch.Tensor] = []
    for i in range(N_sigmas):
        sigma_1 = gaussian_blurs[0:1, i::N_sigmas]
        for j in range(i):
            sigma_2 = gaussian_blurs[0:1, j::N_sigmas]
            diff_list.append(sigma_2 - sigma_1)
    dogs = torch.cat(diff_list, dim=1)
    return dogs


def get_membrane_proj_kernel(
    device: torch.device,
    dtype: torch.dtype,
    n_channels: int,
    membrane_patch_size: int = 17,
    membrane_thickness: int = 1,
    angle_increment_deg: int = 6,
) -> torch.Tensor:
    kernel = np.zeros((membrane_patch_size, membrane_patch_size))
    x0 = membrane_patch_size // 2 - membrane_thickness // 2
    x1 = 1 + membrane_patch_size // 2 + membrane_thickness // 2
    kernel[:, x0:x1] = 1

    all_kernels = [
        np.rint(rotate(kernel, angle, reshape=False))
        for angle in range(0, 180, angle_increment_deg)
    ]
    kernel_np = np.stack(all_kernels)
    kernel_torch = torch.tensor(
        kernel_np, device=device, dtype=dtype, requires_grad=False
    )
    filters = kernel_torch.unsqueeze(1)
    filters = torch.tile(filters, (n_channels, 1, 1, 1))

    return filters


def membrane_projections(
    img: torch.Tensor, kernel: torch.Tensor, N_ch: int
) -> torch.Tensor:
    # TODO: most other gpu filters work with N-channel imgs s.t slicing the output stack
    # by every N you'll get the stack for the Nth channel - this does not work like that
    projs = convolve(img, kernel, False)

    projs_per_ch = projs.shape[1] // N_ch

    projections_out: list[torch.Tensor] = []
    for i in range(N_ch):
        sum_proj = torch.sum(
            projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1
        )
        mean_proj = torch.mean(
            projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1
        )
        std_proj = torch.std(
            projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1
        )
        median_proj, _ = torch.median(
            projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1
        )
        max_proj = torch.amax(
            projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1
        )
        min_proj = torch.amin(
            projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1
        )
        projections_out += [
            mean_proj,
            max_proj,
            min_proj,
            sum_proj,
            std_proj,
            median_proj,
        ]

    return torch.stack(projections_out, dim=1)


def zero_scale_filters(
    img: torch.Tensor,
    sobel_kernel: torch.Tensor,
    sobel_squared_kernel: torch.Tensor,
    sobel_filter: bool = True,
    hessian_filter: bool = True,
    add_mod_trace: bool = True,
) -> list[torch.Tensor]:
    """Weka *always* adds the original image, and if computing edgees and/or hessian,
    adds those for sigma=0. This function does that."""
    out_filtered: list[torch.Tensor] = [img]
    edges = convolve(img, sobel_kernel, True)
    if sobel_filter:
        out_filtered.append(get_gradient_mag(edges))
    if hessian_filter:
        hessian = singescale_hessian(edges, sobel_squared_kernel, add_mod_trace)
        out_filtered.append(hessian)
    return out_filtered


@torch.no_grad()
def multiscale_features_gpu(
    raw_img: torch.Tensor,
    config: FeatureConfig,
    reshape_squeeze: bool = True,
) -> torch.Tensor:
    dtype = raw_img.dtype
    _, C, _, _ = raw_img.shape
    amax = torch.amax(raw_img)
    converted_img = (raw_img * (1 / amax)).to(dtype)

    device = raw_img.device
    mult = 0.4 if config.add_weka_sigma_multiplier else 1
    gauss_kernel = get_multiscale_gaussian_kernel(device, dtype, config.sigmas, C, mult)
    sobel_kernel = get_sobel_kernel(device, dtype, C)
    sobel_squared = get_sobel_kernel(device, dtype, 2 * C)

    membrane_kernel = get_membrane_proj_kernel(
        device, dtype, C, config.membrane_patch_size, config.membrane_thickness
    )

    gaussian_blurs = convolve(converted_img, gauss_kernel, norm=False)

    features: list[torch.Tensor]
    if config.add_zero_scale_features:
        features = zero_scale_filters(
            converted_img,
            sobel_kernel,
            sobel_squared,
            config.sobel_filter,
            config.hessian_filter,
            config.add_mod_trace_det_hessian,
        )
    else:
        features = []

    N_sigmas = len(config.sigmas)
    for i, sigma in enumerate(config.sigmas):
        s = int(sigma)
        blurred = gaussian_blurs[0:1, i::N_sigmas]
        edges = convolve(blurred, sobel_kernel, True)
        if config.gaussian_blur:
            features.append(blurred)
        if config.sobel_filter:
            features.append(get_gradient_mag(edges))
        if config.hessian_filter:
            hess = singescale_hessian(
                edges, sobel_squared, config.add_mod_trace_det_hessian
            )
            features.append(hess)

        if config.mean:
            features.append(singlescale_mean(raw_img, s))
        if config.minimum:
            features.append(singlescale_minimum(raw_img, s))
        if config.maximum:
            features.append(singlescale_maximum(raw_img, s))
        if config.median:
            features.append(singlescale_median(raw_img, s))
        if config.laplacian:
            features.append(singlescale_laplacian(blurred, s))

    if config.difference_of_gaussians:
        features.append(difference_of_gaussians(gaussian_blurs, N_sigmas))
    if config.membrane_projections:
        projections = membrane_projections(converted_img, membrane_kernel, C)
        features.append(projections)
    if config.bilateral:
        features.append(bilateral(raw_img))

    features_out = torch.cat(features, dim=1)
    if config.cast_to == "f16":
        features_out = features_out.to(torch.float16)
    elif config.cast_to == "f64":
        features_out = features_out.to(torch.float32)
    else:
        features_out = features_out.to(torch.float64)

    if reshape_squeeze:
        features_out = torch.squeeze(features_out, 0)
        features_out = torch.permute(features_out, (1, 2, 0))
    return features_out


torch.cuda.empty_cache()
device = torch.device("cuda:0")
if __name__ == "__main__":
    cfg = FeatureConfig(
        name="default",
        cast_to="f16",
        add_zero_scale_features=True,
        hessian_filter=True,
        sobel_filter=True,
        membrane_projections=True,
        difference_of_gaussians=True,
        mean=True,
        minimum=True,
        maximum=True,
        use_gpu=True,
    )
    n_ch = 3
    img = torch.rand(
        (1, n_ch, 400, 400), device=device, dtype=torch.float16, requires_grad=False
    )

    start = time()
    torch.cuda.synchronize()
    feats = multiscale_features_gpu(img, cfg)
    feats_np = feats.cpu().numpy()
    torch.cuda.synchronize()
    end = time()
    print(f"{feats.shape} in {end - start:.4f}s")

    img_l = img[0:1, 0:1, :, :]

    feats_l = multiscale_features_gpu(img_l, cfg)
    feats_l_np = feats_l.cpu().numpy()
    print(f"{feats_l_np.shape}")
