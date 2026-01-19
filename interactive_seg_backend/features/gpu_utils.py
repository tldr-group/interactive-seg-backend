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
