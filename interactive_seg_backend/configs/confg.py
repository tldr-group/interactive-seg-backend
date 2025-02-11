from numpy import log2, logspace
from dataclasses import dataclass, fields, Field, asdict
from json import dumps
from typing import Any

GPU_DISALLOWED_FEATURES: list[str] = [
    "hessian_filter",
    "minimum",
    "maximum",
    "structure_tensor_eigvals",
]


@dataclass
class FeatureConfig:
    # gaussian blur with std=$sigma
    gaussian_blur: bool = True
    # gradient magnitude (of gaussian blur $sigma)
    sobel_filter: bool = True
    # hessian (of gaussian blur $sigma) - either eigs OR eigs + mod, trace, det
    hessian_filter: bool = True
    # difference of all gaussians based on set $sigmas
    difference_of_gaussians: bool = True
    # membrane = convolve img with N pixel line kernel rotated R times
    membrane_projections: bool = True
    # mean of neighbours in $sigma radius
    mean: bool = False
    # minimum of neighbours in $sigma radius
    minimum: bool = False
    # maximum of neighbours in $sigma radius
    maximum: bool = False
    # median of neighbours in $sigma radius
    median: bool = False
    # bilateral: mean of pixels with certain greyscale value within certain radius
    bilateral: bool = False
    # laplacian edge detection (of gaussian blur $sigma)
    laplacian: bool = False

    structure_tensor_eigvals: bool = False

    add_mod_trace_det_hessian: bool = True

    membrane_thickness: int = 1
    membrane_patch_size: int = 15

    # apply features to unblurred img
    add_zero_scale_features: bool = True
    min_sigma: float = 1.0
    max_sigma: float = 16.0
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)

    use_gpu: bool = False

    def _check_if_filters_allowed_with_gpu(self) -> None:
        cls_fields: tuple[Field[Any], ...] = fields(self.__class__)
        for field in cls_fields:
            name = field.name
            is_disallowed = name in GPU_DISALLOWED_FEATURES
            if is_disallowed:
                field_val = self.__getattribute__(name)
                if field_val is True:
                    raise Exception(f"Using a CPU only feature, `{name}` in GPU mode!")

    def __post_init__(self) -> None:
        assert self.min_sigma >= 0, "min_sigma must be greater than (or equal to) 0"
        assert self.max_sigma <= 64, "max_sigma must be less than (or equal to) 64"

        if self.min_sigma != 1.0 or self.max_sigma != 16.0:
            # update sigmas based on min/max if changed
            # NB: to set sigmas explicitly, set it in init and don't adjust min/max
            log_min: float = log2(self.min_sigma)
            log_max: float = log2(self.max_sigma)
            num_sigma = int(log_max - log_min + 1)
            sigmas: tuple[float, ...] = tuple(
                logspace(
                    log_min,
                    log_max,
                    num=num_sigma,
                    base=2,
                    endpoint=True,
                )
            )
            self.sigmas = sigmas

        assert self.membrane_thickness >= 1, (
            "membrane_thickness must be greater than (or equal to) 0"
        )
        assert self.membrane_patch_size >= 3, (
            "membrane_patch_size must be greater than (or equal to) 3"
        )
        if self.use_gpu:
            self._check_if_filters_allowed_with_gpu()

    def __repr__(self) -> str:
        to_stringify = asdict(self)
        out_str = "FEATURE CONFIG: \n" + dumps(
            to_stringify, ensure_ascii=True, indent=2
        )
        return out_str


# @dataclass
# class TrainConfig:
if __name__ == "__main__":
    c = FeatureConfig(sigmas=(1.0, 1.5, 2.0))
    print(c)
