import numpy as np
from typing import Literal

from interactive_seg_backend.core import featurise_, train_and_apply_
from interactive_seg_backend.features import concat_feats
from interactive_seg_backend.features.gpu_utils import transfer_from_gpu
from interactive_seg_backend.configs import (
    AnyArr,
    NPFloatArray,
    NPUIntArray,
    TrainingConfig,
)

AutocontextType = Literal[
    "simple",
    "original",
    "ilastik",
    "autocontext_original",
    "autocontext_ilastik",
]

CARDINAL_DIRECTIONS: list[tuple[int, int]] = [
    (-1, 0),  # Up
    (1, 0),  # Down
    (0, -1),  # Left
    (0, 1),  # Right
]

DIAGONAL_DIRECTIONS: list[tuple[int, int]] = [
    (-1, -1),  # Up-Left
    (-1, 1),  # Up-Right
    (1, -1),  # Down-Left
    (1, 1),  # Down-Right
]


def get_ray_directions(n_rays: Literal[4, 8] = 4) -> list[tuple[int, int]]:
    if n_rays == 4:
        return list(CARDINAL_DIRECTIONS)
    elif n_rays == 8:
        return list(CARDINAL_DIRECTIONS) + list(DIAGONAL_DIRECTIONS)
    else:
        raise ValueError(f"Invalid autocontext_n_rays: {n_rays}. Must be 4 or 8.")


def compute_ray_autocontext_features(
    probs: np.ndarray,
    distances: list[int],
    n_rays: Literal[4, 8] = 4,
) -> np.ndarray:
    """Compute ray context features by rolling probability maps along rays at set distances.

    Args:
        probs (np.ndarray): (H, W, C) or (H, W) array of class probabilities.
        distances (list[int]): list of pixel distances along rays.
        n_rays (Literal[4, 8]): 4 for cardinal directions, 8 to include diagonals.

    Returns:
        np.ndarray: (H, W, n_rays * len(distances) * C) array of rolled probability maps.
    """
    if probs.ndim == 2:
        probs_3d = np.expand_dims(probs, -1)
    else:
        probs_3d = probs

    directions = get_ray_directions(n_rays)
    if not distances:
        raise ValueError("autocontext_distances must not be empty.")

    sorted_distances = sorted(distances)
    rolled_maps: list[np.ndarray] = []

    for dy, dx in directions:
        cur_prob = probs_3d
        prev_d = 0
        for d in sorted_distances:
            delta_d = d - prev_d
            if delta_d != 0:
                cur_prob = np.roll(cur_prob, shift=(dy * delta_d, dx * delta_d), axis=(0, 1))
            rolled_maps.append(cur_prob)
            prev_d = d

    return np.concatenate(rolled_maps, axis=-1)


def autocontext_features(
    image: NPFloatArray,
    labels: NPUIntArray,
    train_cfg: TrainingConfig,
    original_feats: NPFloatArray | None = None,
    original_probs: NPFloatArray | None = None,
) -> AnyArr:
    if original_feats is None:
        feats = featurise_(image, train_cfg.feature_config, train_cfg.preprocessing, train_cfg.use_gpu)
    else:
        feats = original_feats

    if original_probs is None:
        _, probs, _ = train_and_apply_(feats, labels, train_cfg)
    else:
        probs = original_probs

    mode = train_cfg.autocontext_type

    if mode == "simple":
        return concat_feats(feats, probs)
    elif mode in ("original", "autocontext_original"):
        probs_np = transfer_from_gpu(probs) if not isinstance(probs, np.ndarray) else probs
        ray_feats = compute_ray_autocontext_features(
            probs_np,
            distances=train_cfg.autocontext_distances,
            n_rays=train_cfg.autocontext_n_rays,
        )
        return concat_feats(feats, ray_feats)
    elif mode in ("ilastik", "autocontext_ilastik"):
        prob_feats = featurise_(probs, train_cfg.feature_config, train_cfg.preprocessing, train_cfg.use_gpu)
        return concat_feats(feats, prob_feats)
    else:
        raise ValueError(f"Unknown autocontext mode: {mode}")
