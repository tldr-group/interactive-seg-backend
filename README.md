# interactive-seg-backend

[![CI](https://github.com/tldr-group/interactive-seg-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/tldr-group/interactive-seg-backend/actions/workflows/ci.yml)
<a href="https://opensource.org/licenses/MIT">
            <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT LICENSE">
</a>

Generic backend for interactive feature-based segmentation in python with optional CUDA featurisation. Designed to be easy to use & extend. Check out the [`examples/`](examples/) to get started, or [`interactive-seg-gui`](https://github.com/tldr-group/interactive-seg-gui) for a low/no-install GUI which uses this backend.

## Contents
- [Introduction](#what-is-interactive-segmentation)
- [Minimal example](#minimal-example)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Benchmark](#benchmark)
- [Tests](#tests)
- [Contributing](#contributing)
- [Citation](#citation)
- [References](#references)

## What is interactive segmentation?

Interactive or trainable segmentation trains a lightweight classifier (such as a random forest or XGBoost) to map from features that describe an image to arbritrary, user-drawn labels. It is a popular segmentation technique in biological or materials image analysis for several reasons: it is fast, it is flexible, it requires much less training data than a CNN but can segment more complex structures than, say, thresholding. 

Popular examples include [trainable weka segmentation](https://imagej.net/plugins/tws/), [ilastik](https://www.ilastik.org/) and [napari-apoc](github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification) [1-3].
These tools typically come bundled with a GUI, because this is the best way to add labels, see the result, and correct mistakes by adding more labels.
They also tend to be difficult to extend, either becuase they're already a plugin, or are written in (sensible) languages like C++ or Java. My need to rapidly prototype and test additions to the workflow (especially integration with the pytorch ecosystem) across various projects led to me developing this package. My goal was to create an extensible headerless library, one which could easily integrate with a GUI but that could operate with out (for batch processing etc). I also wanted to add various improvements people have added over the years: autocontext [4], domain-inspired losses [5] and Conditional Random Field (CRF) post-processing [6].

## Minimal example

```python
from interactive_seg_backend.file_handling import load_image, load_labels
from interactive_seg_backend import featurise, TrainingConfig, FeatureConfig, train_and_apply
# setup configs
feature_config = FeatureConfig()
training_config = TrainingConfig(feature_config=feature_config, classifier='random_forest')
# load in data - assumes labels stored as (H,W) uint tiff where 0=unlabelled pixels 
image = load_image('path/to/your/image.png')
labels = load_labels('path/to/your/labels.tiff')
# compute features & do end-to-end training and applying
features = featurise(image, training_config)
segmentation, probabilities, trained_classifier = train_and_apply(features, labels, training_config)
```

## Project structure

This library has been designed to be extensible and flexible, favouring typed pure functions that can be dropped into different workflows. In short: which features are computed for an image are defined in a `FeatureConfig()`, which lives inside a `TrainingConfig()` that contains additional information about which classifier, post-processing etc to use. This `FeatureConfig()` determines the filters called and their length-scales in `multiscale_classical_cpu.py`. This returns a (H,W,N_features) feature stack which is used alongside supplied user labels in `core.py` to train a `Classifier()`. The trained `Classifier()` can then be used in `apply()` to predict classes for unlabelled pixels.

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

One-off:
```bash
mkdir tmp
python -m cProfile -s tottime interactive_seg_backend/features/multiscale_classical_cpu.py > tmp/bench.txt
```

Comparison to Weka for featurising / applying on different sized images. To run the Weka beanshell scripts (`*.bsh`), you'll need to use ImageJ/FIJI's script editor (with BeanShell support). For me this is `Help>Examples>BeanShell>Sphere`, which opens a small popup. Then navigate to `File>New`, which opens the editor proper. You can then open the scripts and use the `Run` button.


| Step | Size | Weka | Ours (CPU) | Ours (GPU)* |
| ---- | ---- | ---- | ---- | ---- |
| Featurising | (512, 512) | 1034 ± 100 ms | 489 ± 109 ms | 27 ± 3 ms
| Featurising | (1536, 1536) | 9796 ± 172 ms | 4810 ± 205 ms | 159 ± 4 ms
| Applying | (512, 512) | 828 ± 166 ms | 417 ± 34 ms | 15 ± 0.4 ms |
| Applying | (1536, 1536) | 7191 ± 580 ms | 4687 ± 300 ms | 132 ± 3 ms 


*GPU times don't count passing features -> CPU transfer & GPU apply uses XGBoost, which is faster in general than random-forest

**Applying times measured on 3-phase SEM image with complex structure - this should produce a representatively complex tree structure

***Measured over 10 repeats

## Tests

Requires the pytest package (`pip install '.[dev]'`)

```bash
mkdir tests/out
# grab the reference feature stack:
curl -o tests/data/feature-stack.tif https://sambasegment.blob.core.windows.net/resources/isb_test_data/feature-stack.tif
pytest -s
# or, `uv run pytest -s `
```

## Contributing

Contributions are always welcome! Just create a branch, write the feature and open a pull-request to main - this should trigger the CI (which runs test and code linting). The CI needs to pass before the branch can be merged. If you are adding a feature, please add commensurate tests to `tests/`. Examples of things that would be good additions:

- More features! Stuff like [Gabor](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/TRAPP1/filter.html), [Kuwahara](https://blog.maximeheckel.com/posts/on-crafting-painterly-shaders/), [Lipschitz](https://imagej.net/ij/plugins/lipschitz/) or [Entropy](https://imagej.net/plugins/tws/#:~:text=Entropy) filters, as well as GPU implementations. Approximations are a-okay!
- More classifiers, such as [Generalized Additive Models](https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html), Gaussian process classifiers, RBF SVMs etc. Also of interest are GPU versions of these classifers.
- More segmentation post-processing approaches i.e. min size / mean curvature / ... filtering.
- (Pretty tough one) For small images, most the GPU cost is in data transfer - I current do img CPU -> feats CUDA -> feats CPU -> feats CuPy -> pred CPU. Cutting out the intermediate feats CUDA -> feats CPU could save some time. Where I got held up was making sure the typing story worked with cpu only builds once the array type was generalised to include torch tenors - this could be avoided with a separate API but I'd prefer to keep everything unified.

## Citation

## References

1. I.Arganda-Carreras, *et al.*, "Trainable Weka Segmentation: a machine learning tool for microscopy pixel classification", *Bioinformatics*, (2017)
2. S. Berg, *et al.*, "ilastik: interactive machine learning for (bio)image analysis", *Nature Methods*, (2019)
3. R. Haase, *et al.*, "napari-accelerated-pixel-and-object-classification", *GitHub*, (2021)
4. I.Arganda-Carreras, *et al.*, "Trainable Weka Segmentation: a machine learning tool for microscopy pixel classification", *Bioinformatics*, (2017)
5. N. Prakash, *et al.*, "ExpertSegmentation: Segmentation for microscopy with domain-informed targets via custom loss", *Acta Materialia*, (2025)
6. I.Arganda-Carreras, *et al.*, "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials", *Neurips*, (2012)