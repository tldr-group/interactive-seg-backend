import numpy as np
from typing import cast, TYPE_CHECKING, TypeGuard

from interactive_seg_backend.configs.types import Arrlike, AnyArr

try:
    import torch

    torch_imported = True
except ImportError:
    print("GPU dependencies not installed!")
    torch_imported = False
TORCH_AVAILABLE = torch_imported

if TYPE_CHECKING:
    import torch


def prepare_for_gpu(arr: np.ndarray, device: str = "cuda:0", dtype: "torch.dtype" = torch.float32) -> "torch.Tensor":
    ndims = len(arr.shape)
    if ndims == 2:
        arr = np.expand_dims(arr, (0, 1))  # (H, W) -> (1, 1, H, W)
    else:
        channel_idx = np.argmin(arr.shape)
        if channel_idx == ndims - 1:  # (H, W, C) -> (C, H, W)
            arr = np.transpose(arr, (-1, 0, 1))
        arr = np.expand_dims(arr, (0))  # (C, H, W) -> (1, 1, H, W)
    tensor = torch.tensor(arr, device=device, dtype=dtype)
    return tensor


def check_if_tensor(arr: AnyArr) -> TypeGuard["torch.Tensor"]:
    return isinstance(arr, torch.Tensor)


def check_if_numpy(arr: AnyArr) -> TypeGuard[np.ndarray]:
    return isinstance(arr, np.ndarray)


def transfer_from_gpu(tensor: AnyArr, squeeze_batch_dim: bool = False) -> np.ndarray:
    if check_if_tensor(tensor):
        if squeeze_batch_dim:
            tensor = tensor.squeeze(0)
        arr = tensor.detach().cpu().numpy()
    elif check_if_numpy(tensor):
        arr = tensor
    else:
        # should never hit this branch
        raise Exception(f"Invalid type to transfer from GPU: {type(tensor)}")
    return arr


def concat_feats(arr1: Arrlike, arr2: Arrlike) -> Arrlike:
    # (optionally) cast to tensors and concatenate arrays

    arr_1_is_numpy = isinstance(arr1, np.ndarray)
    arr_2_is_numpy = isinstance(arr2, np.ndarray)
    arr_1_is_tensor = not arr_1_is_numpy
    arr_2_is_tensor = not arr_2_is_numpy

    if arr_1_is_tensor and arr_2_is_tensor:
        arr1_, arr2_ = cast("torch.Tensor", arr1), cast("torch.Tensor", arr2)
        res = torch.concatenate((arr1_, arr2_), dim=-1)
    elif arr_1_is_tensor and arr_2_is_numpy:
        arr1_ = cast("torch.Tensor", arr1)
        tensor_2 = torch.tensor(arr2, dtype=arr1_.dtype, device=arr1_.device)
        res = torch.concatenate((arr1_, tensor_2), dim=-1)
    elif arr_1_is_numpy and arr_2_is_tensor:
        arr2_ = cast("torch.Tensor", arr2)
        tensor_1 = torch.tensor(arr1, dtype=arr2_.dtype, device=arr2_.device)
        res = torch.concatenate((tensor_1, arr2_), dim=-1)
    elif arr_1_is_numpy and arr_2_is_numpy:
        res = np.concatenate((arr1, arr2), axis=-1)
    else:
        raise Exception(f"Invalid feat types: {type(arr1)} + {type(arr2)}")

    return cast(Arrlike, res)


# %% ===================================KORNIA FILTERS===================================
"""
The following block is taken from kornia.filters.kernels & kornia.filters

kornia: https://github.com/kornia/kornia
Apache 2.0
"""


def unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)


def get_binary_kernel2d(
    window_size: tuple[int, int] | int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12

    ky, kx = unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = torch.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma**2))
    return gauss / gauss.sum()


def _get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.

    Returns:
        1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples:

        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. Got {}".format(kernel_size))
    window_1d: torch.Tensor = _gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    force_even: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma: gaussian standard deviation in the x and y
         direction.
        force_even: overrides requirement for odd kernel size.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = _get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = _get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    kernel_2d = kernel_2d.to(device=device, dtype=dtype)
    return kernel_2d


def compute_zero_padding(kernel_size: tuple[int, int]) -> tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: list[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]
