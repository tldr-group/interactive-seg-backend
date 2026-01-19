import numpy as np
from typing import TypeVar
import numpy.typing as npt

from typing import TypeAlias, Literal, TYPE_CHECKING

torch_imported = True
try:
    from torch import Tensor
except ImportError:
    torch_imported = False
TORCH_AVAILABLE: bool = torch_imported

if TYPE_CHECKING:
    from torch import Tensor

NPFloatArray: TypeAlias = npt.NDArray[np.floating]
NPIntArray: TypeAlias = npt.NDArray[np.integer]
NPUIntArray: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]

UInt8Arr: TypeAlias = npt.NDArray[np.uint8]
Arr: TypeAlias = NPFloatArray | NPIntArray

AnyArr: TypeAlias = "Arr | Tensor"
Arrlike = TypeVar(
    "Arrlike",
    Arr,
    "Tensor",
    AnyArr,
)

UInt8Arrlike = TypeVar("UInt8Arrlike", npt.NDArray[np.uint8], "Tensor")

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
