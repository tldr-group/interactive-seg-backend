from numpy import log2, logspace
from dataclasses import dataclass, fields, Field, field, asdict
from json import dumps
from typing import Any, Literal, get_args, cast

from .types import (
    PossibleFeatures,
    ClassifierNames,
    Preprocessing,
    HITLStrategy,
    Rules,
)

_FEAT_LIST: list[PossibleFeatures] = get_args(PossibleFeatures)  # type: ignore
FEATURES: set[PossibleFeatures] = {*_FEAT_LIST}
SCALEFREE_FEATS: list[PossibleFeatures] = _FEAT_LIST[-3:]
GPU_DISALLOWED_FEATURES: set[PossibleFeatures] = {
    "structure_tensor_eigvals",
}
strs_per_singlescale_feat: dict[PossibleFeatures, tuple[str, ...]] = {
    "gaussian_blur": ("gaussian_blur",),
    "sobel_filter": ("sobel_filter",),
    "hessian_filter": (
        "hess_eig_1",
        "hess_eig_2",
        "hess_mod",
        "hess_trace",
        "hess_det",
    ),
    "mean": ("mean",),
    "minimum": ("minimum",),
    "maximum": ("maximum",),
    "median": ("median",),
    "laplacian": ("laplacian",),
    "structure_tensor_eigvals": ("structure_eig_1", "structure_eig_2"),
}
# TODO: add json saving / loading
# TODO: add some example JSONs


@dataclass
class FeatureConfig:
    """Set of (classical) features for the image, & the length scales to apply them over."""

    name: str = "default"
    desc: str = "weka-style features"

    cast_to: Literal["f16", "f32", "f64"] = "f16"

    # weka has a 0.4 * multiplier before its $sigma for the gaussian blurs
    add_weka_sigma_multiplier: bool = True
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
        for cls_field in cls_fields:
            name = cls_field.name
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

    def get_filter_strings(self) -> list[str]:
        out: list[str] = []
        cls_fields: tuple[Field[Any], ...] = fields(self.__class__)

        singlescale_feats = list(strs_per_singlescale_feat.keys())

        def _get_feat_name(feat_type: PossibleFeatures, sigma: float) -> list[str]:
            feat_str_li: list[str] = []
            if name == "hessian_filter":
                feat_str_tuple = strs_per_singlescale_feat[feat_type]
                feat_str_li = [f"{val}_σ{sigma}" for val in feat_str_tuple]
                if self.add_mod_trace_det_hessian is False:
                    feat_str_li = feat_str_li[:2]
            elif name in singlescale_feats:
                feat_str_tuple = strs_per_singlescale_feat[feat_type]
                feat_str_li = [f"{val}_σ{sigma}" for val in feat_str_tuple]
            return feat_str_li

        if self.add_zero_scale_features:
            for name in ("gaussian_blur", "sobel_filter", "hessian_filter"):
                is_enabled = self.__getattribute__(name)
                if is_enabled:
                    out += _get_feat_name(name, 0)

        for sigma in self.sigmas:
            for cls_field in cls_fields:
                name = cls_field.name
                if name not in FEATURES:
                    continue
                name = cast(PossibleFeatures, name)

                is_enabled = self.__getattribute__(name)
                if not is_enabled:
                    continue

                if name in singlescale_feats:
                    out += _get_feat_name(name, sigma)

        for name in SCALEFREE_FEATS:
            is_enabled = self.__getattribute__(name)
            if not is_enabled:
                continue

            if name == "difference_of_gaussians":
                for i, sigma_1 in enumerate(self.sigmas):
                    for j in range(i):
                        sigma_2 = self.sigmas[j]
                        out.append(f"DoG_σ{sigma_1}_{sigma_2}")
            elif name == "membrane_projections":
                projs = ("sum", "mean", "std", "median", "max", "min")
                out += [f"membrane_{proj}" for proj in projs]
            elif name == "bilateral":
                for spatial_radius in (5, 10):
                    for value_range in (50, 100):
                        out.append(f"bilateral_σs{spatial_radius}_σv{value_range}")
        return out

    def __repr__(self) -> str:
        to_stringify = asdict(self)
        out_str = f"FEATURE CONFIG: \n`{self.name}`: {self.desc}\n" + dumps(
            to_stringify, ensure_ascii=True, indent=2
        )
        to_stringify.pop("name")
        to_stringify.pop("desc")
        return out_str


@dataclass
class TrainingConfig:
    """Config for end-to-end training: features, classifier, processing, improvements."""

    feature_config: FeatureConfig

    classifier: ClassifierNames = "random_forest"
    # `classifier_params` are any addtional params passed to classifier init (i.e tree_depth etc)
    # we need field(default_factory) as dicts are mutable and therefore can't be dataclass default args
    classifier_params: dict[str, Any] = field(default_factory=dict)

    balance_classes: bool = True
    shuffle_data: bool = True
    n_samples: int = -1

    preprocessing: tuple[Preprocessing] | None = None

    modal_filter: bool = False
    modal_filter_k: int = 2

    autocontext: bool = False
    CRF: bool = False

    HITL: bool = False
    HITL_strategy: HITLStrategy = "wrong"
    rules: tuple[Rules] | None = None

    use_gpu: bool = False

    def __repr__(self) -> str:
        name = self.feature_config.name
        desc = self.feature_config.desc
        to_stringify = asdict(self)
        to_stringify["feature_config"] = f"`{name}`: {desc}"
        out_str = "TRAINING CONFIG: \n" + dumps(
            to_stringify, ensure_ascii=True, indent=2
        )
        return out_str

    def __post_init__(self) -> None:
        if self.balance_classes and self.classifier in (
            "random_forest",
            "logistic_regression",
        ):
            self.classifier_params["class_weight"] = "balanced"


if __name__ == "__main__":
    c = FeatureConfig(
        sigmas=(1.0, 1.5, 2.0),
        bilateral=True,
        laplacian=True,
        add_mod_trace_det_hessian=False,
    )
    foo = c.get_filter_strings()
    print(foo)
    print(c)
    print(" ")
    t = TrainingConfig(c, "xgb")
    print(t)
