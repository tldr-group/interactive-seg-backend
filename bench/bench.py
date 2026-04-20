from IPython.testing.decorators import f
from PIL.ImageFont import load
import numpy as np
from time import time_ns
from torch.cuda import synchronize
from interactive_seg_backend.file_handling import load_image, load_labels
from interactive_seg_backend import featurise, FeatureConfig, TrainingConfig, train, apply, train_and_apply
import argparse

feat_cfg = FeatureConfig()
cpu_cfg = TrainingConfig(feature_config=feat_cfg, classifier_params={"n_estimators": 200, "max_features": 2})
gpu_cfg = TrainingConfig(
    feature_config=FeatureConfig(cast_to="f16"),
    use_gpu=True,
    classifier="xgb",
)


def bench_feats_isb(img: np.ndarray, cfg: TrainingConfig, N: int) -> None:
    featurise(img, cfg)  # warmup
    times = []
    for _ in range(N):
        synchronize()
        start = time_ns()
        featurise(img, cfg)
        synchronize()
        end = time_ns()
        times.append(1000 * (end - start))
    print(f"ISB; Feats for ({img.shape}); GPU={cfg.use_gpu};\nTime: {np.mean(times):.3f} ± {np.std(times):.3f} seconds")


def bench_apply_isb(img: np.ndarray, labels: np.ndarray, cfg: TrainingConfig, N: int) -> None:
    feats = featurise(img, cfg)
    _, _, model = train_and_apply(feats, labels, cfg)

    times = []
    for _ in range(N):
        synchronize()
        start = time_ns()
        apply(model, feats, cfg)
        synchronize()
        end = time_ns()
        times.append(1000 * (end - start))
    print(f"ISB; Apply for ({img.shape}); GPU={cfg.use_gpu};\nTime{np.mean(times):.3f} ± {np.std(times):.3f} seconds")


if __name__ == "__main__":
    """
    uv run bench/bench.py --small_img_path bench/data/0.png \
    --large_img_path bench/data/0_large.png \
    --labels_path bench/data/0_labels.tif \
    --N 10
    """

    parser = argparse.ArgumentParser(description="Benchmark ISB and Weka feature extraction and application.")
    parser.add_argument("--small_img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--large_img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the labels file")
    parser.add_argument("--N", type=int, default=10, help="Number of repetitions for benchmarking")
    args = parser.parse_args()

    small_img_path = args.small_img_path
    large_img_path = args.large_img_path
    labels_path = args.labels_path
    N = args.N

    small_img = load_image(f"{small_img_path}")
    large_img = load_image(f"{large_img_path}")
    labels = load_labels(labels_path)

    bench_feats_isb(small_img, cpu_cfg, N)
    bench_feats_isb(small_img, gpu_cfg, N)
    bench_apply_isb(small_img, labels, cpu_cfg, N)
    bench_apply_isb(small_img, labels, gpu_cfg, N)

    bench_feats_isb(large_img, cpu_cfg, N)
    bench_feats_isb(large_img, gpu_cfg, N)
    bench_apply_isb(large_img, labels, cpu_cfg, N)
    bench_apply_isb(large_img, labels, gpu_cfg, N)
