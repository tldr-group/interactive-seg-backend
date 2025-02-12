"""
for each sigma:
    1) precompute


# gauss, sobel, via kornia / manually
# hessian by double sobel - maybe do gauss, sobel manually s.t can reuse here
# mean either with kornia bilateral or by convolving with (unit kernel / N_px)
# median via kornia
# max & min with max/min pools and stride  = 1(NB this will be square but w/e). min_pool = - maxpool(-x)
# laplace via kornia
# bilateral via kornia
# membrane projections manually
"""

import torch
from torch.nn.functional import conv2d, max_pool2d, avg_pool2d
from kornia.filters import (
    gaussian_blur2d,
    median_blur,
    laplacian,
    bilateral,
    get_gaussian_kernel2d,
)

from time import time

from interactive_seg_backend.configs import FeatureConfig


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
    mult: float = 0.4,
) -> torch.Tensor:
    # get kernel of shape (N_s, max_k, max_k) where max_k is largest (truncated) gaussian kernel
    N = len(sigmas)
    max_s = max(sigmas)
    max_k = 2 * int(max_s * mult) + 1
    filters = torch.zeros((N, 1, max_k, max_k), dtype=dtype, device=device)
    for i, sigma in enumerate(sigmas):
        filters[i, :, :, :] = get_gaussian_kernel2d(
            (max_k, max_k), (sigma, sigma), device=device, dtype=dtype
        )
    filters = torch.tile(filters, (n_channels, 1, 1, 1))
    return filters


def get_sobel_kernel(
    device: torch.device, dtype: torch.dtype, n_channels: int
) -> torch.Tensor:
    g_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device)
    g_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device)

    stacked = torch.stack((g_x, g_y))
    filters = stacked.unsqueeze(1)
    filters = torch.tile(filters, (n_channels, 1, 1, 1))
    return filters


def convolve(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    _, in_ch, _, _ = img.shape
    _, _, kh, kw = kernel.shape
    return conv2d(img, kernel, padding=(kh // 2, kw // 2), stride=1, groups=in_ch)


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
    second_deriv = convolve(dx_dy, sobel_kernel)
    a: torch.Tensor
    b: torch.Tensor
    d: torch.Tensor

    a, d, _, b = second_deriv[0]

    mod = torch.sqrt(a**2 + b**2 + d**2)
    trace = a + d
    det = a * d - b**2
    # orientation_2 = orientation_1 + np.pi / 2
    # eigvals = feature.hessian_matrix_eigvals(H_elems)
    eig1 = trace + torch.sqrt((4 * b**2 + (a - d) ** 2))
    eig2 = trace - torch.sqrt((4 * b**2 + (a - d) ** 2))

    to_stack: tuple[torch.Tensor, ...]
    if return_full:
        to_stack = (eig1 / 2.0, eig2 / 2.0, mod, trace, det)
    else:
        to_stack = (eig1 / 2.0, eig2 / 2.0)
    out = torch.stack(to_stack)
    return out.unsqueeze(0)


if __name__ == "__main__":
    n_ch = 1
    sigmas = (1.0, 2.0, 4.0, 8.0, 16.0)
    gauss = get_multiscale_gaussian_kernel("cuda:0", torch.float32, sigmas, n_ch)
    sobel_kernel = get_sobel_kernel("cuda:0", torch.float32, n_ch)
    sobel_squared = get_sobel_kernel("cuda:0", torch.float32, 2 * n_ch)
    print(gauss.shape)
    print(sobel_kernel.shape)

    img = torch.rand((1, n_ch, 1000, 1000), device="cuda:0", dtype=torch.float32)
    start = time()
    # feats = convolve(img, gauss)
    feats = convolve(img, sobel_kernel)
    hess = singescale_hessian(feats, sobel_squared)
    print(hess.shape)
    end = time()
    print(f"{feats.shape} in {end - start:.4f}s")
