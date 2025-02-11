# interactive-seg-backend
Generic backend for interactive feature-based segmentation in python.

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