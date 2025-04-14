# interactive-seg-backend
Generic backend for interactive feature-based segmentation in python.

## TODO:
- file handling
- data handling
- classifier(s)
- applying: patched, 3D (+ average), all with memory consideration (caching)
- (pre/post) processing
- improvements: fixed vf, autocontext, rules?, crf?

## Installation:

To install locally in editable mode:
```bash
git clone https://github.com/tldr-group/interactive-seg-backend
cd interactive-seg-backend
pip install -e .
```

For GPU-enabled featurising, install with:
```bash
pip install -e '.[gpu]'
```

For development (linters, tests), install with
```bash
pip install -e '.[test]'
```

To get all the optional dependencies at once:
```bash
pip install -e '.[all]'
```

## Benchmark
```bash
python -m cProfile -s tottime multiscale_classical_cpu.py > bench.txt
```

## Tests

Requires the pytest package (`pip install -e '.[test]'`)
```bash
pytest -s
```



## Install locally (i.e for use in GUI dev)
```bash
pip uninstall interactive_seg_backend -y
pip install . --no-cache-dir
```

If offline
```bash
pip uninstall interactive_seg_backend -y
pip install . --no-cache-dir --no-index
```

## Notes:
- your CRF params are always default!!