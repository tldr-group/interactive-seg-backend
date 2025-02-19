import numpy as np
from typing import TypeVar
import numpy.typing as npt
from torch import Tensor
from typing import TypeAlias

FloatArr: TypeAlias = (
    npt.NDArray[np.float16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]
)
IntArr: TypeAlias = (
    npt.NDArray[np.uint8]
    | npt.NDArray[np.uint16]
    | npt.NDArray[np.int32]
    | npt.NDArray[np.int64]
)
Arr: TypeAlias = FloatArr | IntArr
# FloatArr = TypeVar(
#     "FloatArr",
#     npt.NDArray[np.float16],
#     npt.NDArray[np.float32],
#     npt.NDArray[np.float64],
# )
Arrlike = TypeVar(
    "Arrlike",
    npt.NDArray[np.float16],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
    Tensor,
)
