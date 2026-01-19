# interactive-seg-backend

Generic backend for interactive feature-based segmentation in python.

## Installation:

To install:

```bash
git clone https://github.com/tldr-group/interactive-seg-backend
cd interactive-seg-backend
pip install .
# requires some extra build steps
pip install --no-build-isolation "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git"
```

### Pip:

For GPU-enabled featurising (recommended), install with:

```bash
pip install '.[gpu]'
```

For development (linters, tests), install with

```bash
pip install -e '.[lint,test]'
```

To get all the optional dependencies at once:

```bash
pip install '.[all]'
```

### UV:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your shell
```

```bash
uv sync --extra gpu
uv pip install --no-build-isolation "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git"
```

```bash
uv sync --all-extras
uv pip install --no-build-isolation "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git"
```

## Benchmark

```bash
mkdir tmp
python -m cProfile -s tottime interactive_seg_backend/features/multiscale_classical_cpu.py > tmp/bench.txt
```

## Tests

Requires the pytest package (`pip install '.[test]'`)

```bash
mkdir tests/data
curl -o tests/data/feature-stack.tif https://sambasegment.blob.core.windows.net/resources/isb_test_data/feature-stack.tif
pytest -s
```

## Install locally (i.e for dev)

```bash
pip uninstall interactive_seg_backend -y
pip install -e . --no-cache-dir
```

If offline

```bash
pip uninstall interactive_seg_backend -y
pip install . --no-cache-dir --no-index
```

## TODO:

- make typing story more compelling:
  - actually make the main functions able to take in tensors or arrays
  - or give up and make separate functions where necesarry
  - make core vs main distinction make more sense
    - export everything important from main / init
    - pass down things you care about i.e sample weights into train / train and apply
- logging
- try to adapt kornia filter code w/out importing the whole thing ()
- improvements: fixed vf, rules (connectivity), SAM post-proc ?
- docstrings
- pydantic?
