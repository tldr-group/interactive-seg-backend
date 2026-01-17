import numpy as np
from typing import TypeVar
import numpy.typing as npt

from typing import TypeAlias, Literal, TYPE_CHECKING, Protocol, runtime_checkable

torch_imported = True
try:
    from torch import Tensor
except ImportError:
    torch_imported = False
TORCH_AVAILABLE: bool = torch_imported

if TYPE_CHECKING:
    from torch import Tensor


DType = TypeVar("DType", covariant=False)


@runtime_checkable
class Array(Protocol[DType]):
    shape: tuple[int, ...]
    dtype: DType

    def reshape(self, shape: tuple[int, ...]) -> "Array[DType]": ...


class FloatDType: ...


class IntDType: ...


FloatArr_: TypeAlias = Array[FloatDType]
IntArr_: TypeAlias = Array[IntDType]

NPFloatArray: TypeAlias = npt.NDArray[np.floating]
NPIntArray: TypeAlias = npt.NDArray[np.integer]
NPUIntArray: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]

# NPFloatArr: TypeAlias = Array[np.floating]

FloatArr: TypeAlias = npt.NDArray[np.float16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]
UInt8Arr: TypeAlias = npt.NDArray[np.uint8]
IntArr: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.int32] | npt.NDArray[np.int64]
Arr: TypeAlias = FloatArr | IntArr

AnyArr: TypeAlias = Arr | Tensor
Arrlike = TypeVar(
    "Arrlike",
    FloatArr,
    Arr,
    Tensor,
    AnyArr,
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

ClassifierNames = Literal["linear_regression", "logistic_regression", "random_forest", "xgb", "mlp"]
Preprocessing = Literal["denoise", "equalize", "blur"]
Postprocessing = Literal["modal_filter"]
LabellingStrategy = Literal["sparse", "dense", "interfaces"]
HITLStrategy = Literal["wrong", "uncertainty", "representative_weighted"]
Extensions = Literal["autocontext_original", "autocontext_ilastik", "rules_vf", "rules_connectivity"]
Rules = Literal["vf", "connectivity"]

# TODO: define a feature stack dataclass? image h, w, list of features used to generate it, and save / load helpers (i.e as a paged tiff or .pt)
