import pytest
import numpy as np
import numpy.typing as npt
import torch
from math import isclose, pi
from skimage.metrics import mean_squared_error
from tifffile import imread

from interactive_seg_backend.configs import FeatureConfig
import interactive_seg_backend.features.multiscale_classical_cpu as ft
import interactive_seg_backend.features.multiscale_classical_gpu as ft_gpu

# from visualise import plot

np.random.seed(1234521)
SIGMA = 5
FOOTPRINT = ft.make_footprint(sigma=SIGMA)
CIRCLE = np.pad(FOOTPRINT, ((2, 2), (2, 2))).astype(np.float32)
CENTRE = (SIGMA + 2, SIGMA + 2)
CIRCLE_BYTE = (255 * CIRCLE).astype(np.uint8)

DATA = imread("tests/data/0.tif")


def _test_centre_val(filtered_arr: npt.NDArray[np.uint8], val: float | int):
    centre_val = filtered_arr[CENTRE[1], CENTRE[0]]
    assert isclose(centre_val, val, abs_tol=1e-6)


def _test_sum(
    filtered_arr: npt.NDArray[np.float32] | npt.NDArray[np.uint8], val: float | int
):
    assert isclose(np.sum(filtered_arr), val, abs_tol=1e-6)


class TestFeatureCorrectness:
    """Test featurisation functions in features.py."""

    def test_footprint(self) -> None:
        """Footprint test.

        Check if ratio of footprint circle to footprint square is close to pi*r^2/(2*r+1)^2 - obviously
        not exact as we're comparing discrete value to analytic expression.
        """
        sigma = 6
        footprint = ft.make_footprint(sigma=sigma)
        area_circle = np.sum(footprint)
        square_length = footprint.shape[0]
        area_square = square_length**2
        footprint_ratio = area_circle / area_square
        analytic_ratio = pi * sigma**2 / square_length**2
        assert isclose(footprint_ratio, analytic_ratio, rel_tol=0.05)

    def test_sobel(self) -> None:
        """Sobel filter test.

        Perform sobel edge detection on our circle array - should return some outline of the circle.
        Then subtract the original circle and measure circumfrence. If it's similar to the analytic
        circumfrence then ok. (Again discrete != anayltic so have high rel tolerance.)
        """
        filtered = ft.singlescale_edges(CIRCLE)
        circumfrence = np.sum(np.where(filtered > 0, 1, 0))
        assert isclose(circumfrence, 2 * pi * SIGMA, rel_tol=0.5)

    def test_mean(self) -> None:
        """Mean filter test.

        Mean filter w/ footprint of size $SIGMA on our test arr $CIRCLE_BYTE. Because mean filter is same
        size as circle, the mean of the centre should be unity (times a scaling factor) and the sum
        should be unity as well.
        """
        filtered = ft.singlescale_mean(CIRCLE_BYTE, FOOTPRINT)
        _test_centre_val(filtered, 255)

    def test_max(self) -> None:
        """Max filter test.

        Max filter radius $SIGMA on our test arr $CIRCLE_BYTE should be 255 almost everywhere except top
        corners (as they are more than $SIGMA pixels away from disk).
        """
        filtered = ft.singlescale_maximum(CIRCLE_BYTE, FOOTPRINT)
        _test_centre_val(filtered, 255)
        top_left_val = filtered[0, 0]
        assert isclose(top_left_val, 0, abs_tol=1e-6)

    def test_min(self) -> None:
        """Min filter test.

        Min filter radius $SIGMA on our test arr $CIRCLE_BYTE should be 0 almost everywhere except
        centre - so centre value AND sum should equal 255.
        """
        filtered = ft.singlescale_minimum(CIRCLE_BYTE, FOOTPRINT)
        _test_centre_val(filtered, 255)
        _test_sum(filtered, 255)

    def test_median(self) -> None:
        """Median filter test.

        Median filter radius $SIGMA on our test arr $CIRCLE_BYTE should be the circle again but smaller.
        Again centre should be 255 and egdes 0.
        """
        filtered = ft.singlescale_median(CIRCLE_BYTE, FOOTPRINT)
        _test_centre_val(filtered, 255)
        top_left_val = filtered[0, 0]
        assert isclose(top_left_val, 0, abs_tol=1e-6)

    def test_membrane_projection(self) -> None:
        """Membrane projection filter test.

        Membrane projections emphasise lines of similar value pixels. Here we test it on a line of pixels -
        the max value should be in the centre and it should decrease montonically from that.
        """
        line = np.zeros((64, 64), dtype=np.float32)
        line[:, 32] = 1
        z_projs = ft.membrane_projections(line, num_workers=1)
        filtered = z_projs[0]
        prev_val = filtered[32, 32]
        for i in range(1, 5):
            current_val = filtered[32, 32 + i]
            assert current_val < prev_val
            prev_val = current_val

    def test_bilateral(self) -> None:
        """Bilateral filter test.

        Bilateral filter is mean of values in [5, 10] footprint with [50, 100] pixel values of the centre pixel.
        With a bilevel square - one half 20, one half 100 - our bilateral filter with window threhsold 50 won't do
        anything (i.e will average 20s wiwth 20s and 100s with 100s.). However, with window threshold 100, the filter will
        average the 20s and the 100s at the interface and decrease the total value of the array (because 20/100 not symmetric).
        Because our function returns [(5, 50), (5, 100), (10, 50), (10, 100)] where first value is footprint raidus and
        second value is threshold, we check if [0] > [1] as [1] has averaging, we check [0] == [2] as both no averaging
        and we check [0] > [3] as [3] has more averaging.
        """
        bilevel = np.ones((64, 64), dtype=np.uint8) * 20
        bilevel[:, 32:] = 100
        bilaterals = ft.bilateral(bilevel.astype(np.uint8))
        assert np.sum(bilaterals[0]) > np.sum(bilaterals[1])
        assert np.sum(bilaterals[0]) == np.sum(bilaterals[2])
        assert np.sum(bilaterals[0]) > np.sum(bilaterals[3])


def norm(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalise array by subtracting its min then dividing my max of new arr. Works for mixes of positive and negative.

    :param arr: arr to normalise
    :type arr: np.ndarray
    :return: normalised arr
    :rtype: np.ndarray
    """
    offset = arr - np.amin(arr)
    normed = offset / np.amax(offset)
    return np.abs(normed)


def norm_get_mse(
    filter_1: npt.NDArray[np.float32], filter_2: npt.NDArray[np.float32]
) -> float:
    """Normalise both filters (arrays) and get the mse.

    :param filter_1: first arr
    :type filter_1: np.ndarray
    :param filter_2: second arr
    :type filter_2: np.ndarray
    :return: mean squared error between normalised arrs
    :rtype: float
    """
    n1 = norm(filter_1)
    n2 = norm(filter_2)
    return mean_squared_error(n1, n2)


AVG_MSE_CUTOFF = 0.03


def compare_two_stacks(
    stack_1: npt.NDArray[np.float32],
    stack_2: npt.NDArray[np.float32],
    cutoff: float,
    feat_names: list[str],
    skip_feats: list[str] = [],
) -> None:
    checks: list[tuple[str, bool, float]] = []
    for i, feat_name in enumerate(feat_names):
        if feat_name in skip_feats:
            continue
        slice_1 = stack_1[:, :, i]
        slice_2 = stack_2[:, :, i]

        diff = norm_get_mse(slice_1, slice_2)

        passed = diff < cutoff
        checks.append((feat_name, passed, diff))
    passed_all = all([t[1] for t in checks])
    if not passed_all:
        failed = [t for t in checks if t[1] == False]
        print(failed)
        assert False


class TestGPUCPUEquivalence:
    def test_e2e_equiv(self) -> None:
        cfg = FeatureConfig(
            sigmas=(1.0, 2.0, 4.0, 8.0),
            mean=True,
            maximum=True,
            minimum=True,
            median=True,
            bilateral=True,
            add_weka_sigma_multiplier=False,
        )
        feats_cpu: npt.NDArray[np.float32] = ft.multiscale_features(DATA, cfg)

        img_tensor = (
            torch.tensor(DATA, device="cuda:0", dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        assert len(img_tensor.shape) == 4, "need (B,C,H,W) tensor for GPU feats"
        feats_gpu = ft_gpu.multiscale_features_gpu(img_tensor, cfg, torch.float32)
        feats_gpu_np: npt.NDArray[np.float32] = feats_gpu.cpu().numpy()

        assert feats_cpu.shape == feats_gpu.shape, (
            f"{feats_cpu.shape} != {feats_gpu.shape}!"
        )

        feat_names = cfg.get_filter_strings()
        compare_two_stacks(feats_cpu, feats_gpu_np, AVG_MSE_CUTOFF, feat_names)


class TestCPUWekaEquivalence:
    def get_matching_weka_filters(
        self,
        sigmas: tuple[float, ...],
        full_stack: npt.NDArray[np.float32],
        n_feats_out: int,
    ) -> npt.NDArray[np.float32]:
        h, w, _ = full_stack.shape
        out = np.zeros((h, w, n_feats_out), dtype=np.float32)

        new_sigmas = (0, *sigmas)
        filters_per_sigma_zero = 10
        filters_per_sigma_full = 25 - 11
        filters_per_sigma_zero_ours = 7
        filters_per_sigma_full_ours = 11
        scalefree_start_idx = -12

        n_dog = 0
        for i, _ in enumerate(new_sigmas):
            if i > 0:
                weka_init_idx = (
                    (i - 1) * filters_per_sigma_full + filters_per_sigma_zero + n_dog
                )
                ours_init_idx = (
                    i - 1
                ) * filters_per_sigma_full_ours + filters_per_sigma_zero_ours
            else:
                weka_init_idx = 0
                ours_init_idx = 0
            gauss_idx = weka_init_idx
            sobel_idx = weka_init_idx + 1
            hess_eig_1_idx = weka_init_idx + 5
            hess_eig_2_idx = weka_init_idx + 6
            hess_mod_idx = weka_init_idx + 2
            hess_trace_idx = weka_init_idx + 3
            hess_det_idx = weka_init_idx + 4
            singlescale_idxs = [
                gauss_idx,
                sobel_idx,
                hess_eig_1_idx,
                hess_eig_2_idx,
                hess_mod_idx,
                hess_trace_idx,
                hess_det_idx,
            ]

            for k, weka_idx in enumerate(singlescale_idxs):
                out[:, :, ours_init_idx + k] = full_stack[:, :, weka_idx]

            j = 0
            # weka has weird order for last sigma: DoGs first, then min max mean
            if i > 1:
                for j in range(i - 1):  # sigma = 1 has no DoG
                    dog_idx = weka_init_idx + 10 + j
                    ours_dog_idx = scalefree_start_idx + n_dog
                    out[:, :, ours_dog_idx] = full_stack[:, :, dog_idx]
                    n_dog += 1
            if i >= 1:
                for n_avg in range(4):  # mean, min, max, median - same order as ours
                    avg_idx = (
                        weka_init_idx + 10 + (j + 1) + n_avg
                    )  # need to account for their DoG ordering
                    avg_idx_ours = ours_init_idx + len(singlescale_idxs) + n_avg
                    out[:, :, avg_idx_ours] = full_stack[:, :, avg_idx]
        N_MEMBRANE_PROJ = 6
        for mp in range(N_MEMBRANE_PROJ, 0, -1):
            out[:, :, -mp] = full_stack[:, :, -mp]
        return out

    def test_e2e_equiv(self) -> None:
        weka_stack = imread("tests/data/feature-stack.tif").transpose((1, 2, 0))

        cfg = FeatureConfig(
            sigmas=(1.0, 2.0, 4.0, 8.0),
            mean=True,
            maximum=True,
            minimum=True,
            median=True,
            add_weka_sigma_multiplier=True,
            add_zero_scale_features=True,
        )
        n_feats_out = len(cfg.get_filter_strings())
        weka_stack_remapped = self.get_matching_weka_filters(
            cfg.sigmas, weka_stack, n_feats_out
        )
        ours_stack = ft.multiscale_features(DATA, cfg)
        feat_names = cfg.get_filter_strings()

        compare_two_stacks(
            ours_stack,
            weka_stack_remapped,
            AVG_MSE_CUTOFF,
            feat_names,
            ["DoG_σ2.0_1.0"],
        )


if __name__ == "__main__":
    pytest.main()
