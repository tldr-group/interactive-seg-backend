import numpy as np
from typing import TypeVar
import numpy.typing as npt
from torch import Tensor
from typing import TypeAlias, Literal

FloatArr: TypeAlias = (
    npt.NDArray[np.float16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]
)
UInt8Arr: TypeAlias = npt.NDArray[np.uint8]
IntArr: TypeAlias = (
    npt.NDArray[np.uint8]
    | npt.NDArray[np.uint16]
    | npt.NDArray[np.int32]
    | npt.NDArray[np.int64]
)
Arr: TypeAlias = FloatArr | IntArr

Arrlike = TypeVar(
    "Arrlike",
    Arr,
    Tensor,
)

UInt8Arrlike = TypeVar("UInt8Arrlike", npt.NDArray[np.uint8], Tensor)

PossibleFeatures = Literal[
    "gaussian_blur",
    "sobel_filter",
    "hessian_filter",
    "mean",
    "minimum",
    "maximum",
    "median",
    "laplacian",
    "structure_tensor_eigvals",
    "difference_of_gaussians",
    "membrane_projections",
    "bilateral",
]

ClassifierNames = Literal[
    "linear_regression", "logistic_regression", "random_forest", "xgb"
]
Preprocessing = Literal["denoise", "equalize", "blur"]
Postprocessing = Literal["modal_filter"]
LabellingStrategy = Literal["sparse", "dense", "interfaces"]
HITLStrategy = Literal["wrong", "uncertainty", "representative_weighted"]
Rules = Literal["volume_fraction", "connectivity"]

# TODO: define a feature stack dataclass? image h, w, list of features used to generate it, and save / load helpers (i.e as a paged tiff or .pt)
