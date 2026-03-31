# interactive-seg-backend

Generic backend for interactive feature-based segmentation in python with optional CUDA featurisation. Designed to be easy to use & extend. Check out the [`examples/`](examples/) to get started, or [`interactive-seg-gui`](https://github.com/tldr-group/interactive-seg-gui) for a low/no-install GUI which uses this backend.

## Contents
- [Summary](#what-is-interactive-segmentation)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Benchmark](#benchmark)
- [Tests](#tests)
- [Contributing](#contributing)
- [Citation](#citation)
- [References](#references)

## What is interactive segmentation?

## Project structure

This library has been designed to be extensible and flexible, favouring typed pure functions that can be dropped into different workflows. In short: which features are computed for an image are defined in a `FeatureConfig`, which lives inside a `TrainingConfig` that contains additional information about which classifier, post-processing etc to use. This `FeatureConfig` determines the filters called and their length-scales in `multiscale_classical_cpu.py`. This returns a (H,W,N_features) feature stack which is used alongside supplied user labels in `core.py` to train a `Classifier`. 

```bash
examples/ # jupyter notebooks explaing how to run the library
interactive_seg_backend/
├─ classifiers/
│  ├─ base.py # abstract base class (to implement custom classifiers)
│  ├─ sklearn.py # sklearn classifiers: random forest, linear regression, etc
│  ├─ xgb.py
├─ configs/
│  ├─ configs.py # definitions of FeatureConfigs and TrainingConfigs - important wrapper classes
├─ extensions/ # various literature additions to the interactive segmentation workflow
│  ├─ autocontext.py
│  ├─ crf.py
│  ├─ expertseg.py
├─ features/ # multiscale feature extraction
│  ├─ multiscale_classical_cpu.py
│  ├─ multiscale_classical_gpu.py
├─ file_handling/
├─ processing/ # pre and post-processing options
├─ core.py # loading, shuffling, sampling, training, applying functions  
├─ main.py # wrappers that combine core and extension functions
├─ utils.py # plotting utils
tests/ # unit and integration tests
```

## Installation

To install:

```bash
git clone https://github.com/tldr-group/interactive-seg-backend
cd interactive-seg-backend
pip install . '.[cpu]'
```

### Pip:

For the CPU-only version of the package (lighter), install with:
```bash
pip install '.[cpu]'
```
Note: you *must* do this to use xgboost classifiers.

For GPU-enabled featurising (recommended), install with:

```bash
pip install '.[gpu]'
```

For development (linters, tests, notebooks), install with

```bash
pip install -e '.[cpu,dev]'
```

### uv (recommended):

uv is a fast, modern package manager from the ruff developers, which supports reproducible builds via a `uv.lock` file (similar to poetry).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your shell
```

CPU-only:
```bash
uv sync --extra cpu
```

```bash
uv sync --extra gpu --extra dev
```


## Benchmark

```bash
mkdir tmp
python -m cProfile -s tottime interactive_seg_backend/features/multiscale_classical_cpu.py > tmp/bench.txt
```

## Tests

Requires the pytest package (`pip install '.[dev]'`)

```bash
mkdir tests/out
# grab the reference feature stack:
curl -o tests/data/feature-stack.tif https://sambasegment.blob.core.windows.net/resources/isb_test_data/feature-stack.tif
pytest -s
```

## Contributing

## Citation

## References