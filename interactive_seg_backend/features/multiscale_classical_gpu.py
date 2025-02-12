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
    sobel,
    median_blur,
    laplacian,
    bilateral,
    get_gaussian_kernel2d,
)

from time import time


def singlescale_gaussian(
    img: torch.Tensor, sigma: int, mult: float = 1.0
) -> torch.Tensor:
    s = int(mult * sigma)
    out = gaussian_blur2d(img, kernel_size=(2 * s + 1, 2 * s + 1), sigma=(s, s))
    return out


def get_multiscale_gaussian_kernel(
    device: torch.device,
    dtype: torch.dtype,
    sigmas: list[int],
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
    filters = torch.tile(filters, (n_ch, 1, 1, 1))
    return filters


def get_sobel_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device)
    g_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device)

    stacked = torch.stack((g_x, g_y))
    return stacked.unsqueeze(1)


def convolve(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    _, C, _, _ = img.shape
    _, _, kh, kw = kernel.shape
    return conv2d(img, kernel, padding=(kh // 2, kw // 2), stride=1, groups=C)


if __name__ == "__main__":
    n_ch = 1
    sigmas = (1.0, 2.0, 4.0, 8.0, 16.0)
    gauss = get_multiscale_gaussian_kernel("cuda:0", torch.float32, sigmas)
    sobel = get_sobel_kernels("cuda:0", torch.float32)
    print(sobel.shape)
    print(sobel)
    print(gauss.shape)
    img = torch.rand((1, n_ch, 1000, 1000), device="cuda:0", dtype=torch.float32)
    start = time()
    feats = convolve(img, gauss)
    end = time()
    print(f"{feats.shape} in {end - start:.4f}s")
