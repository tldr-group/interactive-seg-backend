import numpy as np
from typing import TypeVar
import numpy.typing as npt
from torch import Tensor

FloatArr = TypeVar(
    "FloatArr",
    npt.NDArray[np.float16],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
)
Arrlike = TypeVar(
    "Arrlike",
    npt.NDArray[np.float16],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
    Tensor,
)
