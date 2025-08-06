# interactive-seg-backend

Generic backend for interactive feature-based segmentation in python.

## Installation:

To install:

```bash
git clone https://github.com/tldr-group/interactive-seg-backend
cd interactive-seg-backend
pip install .
```

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

## Benchmark

```bash
python -m cProfile -s tottime multiscale_classical_cpu.py > bench.txt
```

## Tests

Requires the pytest package (`pip install '.[test]'`)

```bash
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

- applying: patched, 3D (+ average), all with memory consideration (caching)
- improvements: fixed vf, autocontext, rules?, crf?
- make CPU version still work with conditional imports and string quote types
- make typing story more compelling:

  - actually make the main functions able to take in tensors or arrays
  - make core vs main distinction make more sense
    - export everything important from main / init

- docstrings
