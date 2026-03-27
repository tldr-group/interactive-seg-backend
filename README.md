# interactive-seg-backend

Generic backend for interactive feature-based segmentation in python.

## Installation:

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

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your shell
```

CPU-only:
```bash
uv sync --extra cpu
```

```bash
uv sync --extra gpu
```


## Benchmark

```bash
mkdir tmp
python -m cProfile -s tottime interactive_seg_backend/features/multiscale_classical_cpu.py > tmp/bench.txt
```

## Tests

Requires the pytest package (`pip install '.[dev]'`)

```bash
mkdir tests/data
# grab the reference feature stack:
curl -o tests/data/feature-stack.tif https://sambasegment.blob.core.windows.net/resources/isb_test_data/feature-stack.tif
pytest -s
```

## Install locally (i.e for dev)

```bash
pip uninstall interactive_seg_backend -y
pip install -e . '.[cpu, dev]' --no-cache-dir
```

If offline

```bash
pip uninstall interactive_seg_backend -y
pip install . '.[cpu, dev]' --no-cache-dir --no-index
```
