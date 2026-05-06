# patomorphology

AI pipeline for predicting cancer changes in skin histopathology (H&E) whole-slide images.

## Setup

```bash
uv sync
```

This creates `.venv/` and installs all dependencies (including the dev group).

## Run

```bash
uv run python -c "from pato import __version__; print(__version__)"
```

## Tests

```bash
uv run pytest
```

## Layout

```
.
├── data/             # datasets (gitignored)
├── notebooks/        # Jupyter notebooks
├── src/pato/         # main package
│   ├── config.py     # pydantic-based settings
│   ├── data/         # data loading / preprocessing
│   ├── models/       # model definitions
│   └── pipeline/     # training / inference pipelines
└── tests/            # pytest tests
```
