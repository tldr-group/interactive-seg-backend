import numpy as np
import numpy.typing as npt
import torch
from random import shuffle

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tifffile import imread

from interactive_seg_backend.configs import FeatureConfig
from interactive_seg_backend.features import (
    multiscale_features,
    multiscale_features_gpu,
)

from typing import Any


def plot_single_axis(
    fig,
    ax,
    data: np.ndarray,
    title: str,
    ylabel: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fontsize = 20
    if vmin is not None and vmax is not None:
        mappable = ax.imshow(data, cmap="plasma", vmin=vmin, vmax=vmax)
    else:
        mappable = ax.imshow(data, cmap="plasma")
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    fig.colorbar(
        mappable,
        ax=ax,
    )


def setup_figure(n_cols: int, n_rows: int = 2, add_diff_hist: bool = False) -> tuple[Figure, npt.NDArray[Any]]:
    axs: npt.NDArray[Any]
    if add_diff_hist:
        n_rows += 1
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows)
    fig.set_size_inches((4.75 * n_cols), (4.5 * n_rows))
    return fig, axs


def plot(
    cpu_feats: np.ndarray,
    gpu_feats: np.ndarray,
    cfg: FeatureConfig,
    add_diff: bool = False,
    n_subsample: int = -1,
    labels: tuple[str, ...] = ("CPU", "GPU"),
) -> None:
    feat_names = cfg.get_filter_strings()

    inds = list(range(0, len(feat_names)))
    if n_subsample > 0:
        shuffle(inds)
        sample_inds = inds[:n_subsample]
    else:
        sample_inds = inds
    feat_names = [feat_names[ind] for ind in sample_inds]
    print(feat_names)

    fig, axs = setup_figure(len(feat_names), add_diff_hist=add_diff)

    for i, (feat_idx, name) in enumerate(zip(sample_inds, feat_names)):
        arrs = (cpu_feats[:, :, feat_idx], gpu_feats[:, :, feat_idx])
        # vmin, vmax = np.amin(arrs[0]), np.amax(arrs[0])
        for j, arr in enumerate(arrs):
            if i == 0 and j == 0:
                label = labels[0]
            elif i == 0 and j == 1:
                label = labels[1]
            else:
                label = None

            plot_single_axis(
                fig,
                axs[j, i],
                arr,
                name,
                label,
            )  # vmin, vmax
        if add_diff:
            h, w = arrs[0].shape
            diffs = np.abs((arrs[0] - arrs[1])).reshape(h * w)
            try:
                axs[-1, i].hist(diffs, density=True, bins=100)
            except ValueError:
                pass
    plt.tight_layout()
    plt.savefig("tests/out/vis.png")
    plt.close()


if __name__ == "__main__":
    DATA = imread("tests/data/0.tif")
    ADD_DIFF_HIST = True

    cfg = FeatureConfig(
        sigmas=(1.0, 4.0, 16.0),
        mean=True,
        maximum=True,
        minimum=True,
        laplacian=True,
        add_weka_sigma_multiplier=False,
    )
    feats_cpu = multiscale_features(DATA, cfg)

    img_tensor = torch.tensor(DATA, device="cuda:0", dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    feats_gpu = multiscale_features_gpu(img_tensor, cfg)
    feats_gpu_np: np.ndarray = feats_gpu.cpu().numpy()

    plot(feats_cpu, feats_gpu_np, cfg, ADD_DIFF_HIST, 10)
