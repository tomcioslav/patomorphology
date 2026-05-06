# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

`patomorphology` — an AI pipeline that trains a model to predict cancer changes in pathology images. The Python package itself is named **`pato`** and lives under `src/pato/`.

## Toolchain

- **Package manager:** `uv` (pinned via `pyproject.toml` + `uv.lock`). Always use `uv` — never call `pip` or `python` directly.
- **Python:** 3.12 (pinned in `.python-version`).
- **Virtual environment:** `.venv/` in the repo root, managed by `uv sync`. Do not create or manage venvs by hand.
- **Models:** Use **Pydantic v2** (`BaseModel`) for all data classes / DTOs / configuration. Do **not** use `dataclasses.dataclass`. Application/runtime configuration uses `pydantic-settings` (`BaseSettings`).
- **Build backend:** `uv_build` (declared in `pyproject.toml`). Source layout is `src/`.

## Common commands

```bash
uv sync                                  # install / refresh deps and the venv
uv add <pkg>                             # add a runtime dependency
uv add --group dev <pkg>                 # add a dev dependency
uv run python -m <module>                # run code inside the project venv
uv run pytest                            # run tests
uv run ruff check .                      # lint
uv run ruff format .                     # format
uv run jupyter lab                       # start Jupyter for notebooks/
```

## Repository layout

```
.
├── .python-version          # 3.12
├── .gitignore               # ignores .venv, data/, model artifacts, etc.
├── .env.example             # template for PATO_* env vars
├── pyproject.toml           # project metadata + deps
├── uv.lock                  # resolved lockfile (committed)
├── README.md
├── CLAUDE.md                # this file
├── data/                    # datasets — gitignored except README + .gitkeep
├── notebooks/               # Jupyter notebooks
├── src/pato/                # main package
│   ├── __init__.py          # exposes __version__
│   ├── config.py            # pydantic Settings + PathsConfig
│   ├── data/                # data loading / preprocessing (empty)
│   ├── models/              # model definitions (empty)
│   └── pipeline/            # training / inference orchestration (empty)
└── tests/                   # pytest tests
```

`data/`, `src/pato/data/`, `src/pato/models/`, `src/pato/pipeline/`, and `tests/` currently hold only `__init__.py` / `.gitkeep` placeholders — they are scaffolding to fill in as the pipeline grows.

## Conventions

- **Configuration:** read settings via `pato.config.Settings()`. Override with `PATO_`-prefixed env vars or a `.env` file. Nested fields use double-underscore (e.g. `PATO_PATHS__DATA=/abs/path`).
- **Data classes:** every structured object that crosses a module boundary (configs, sample records, model outputs, pipeline I/O) is a `pydantic.BaseModel`. No `dataclasses`, no `TypedDict` for data.
- **Imports:** absolute (`from pato.config import Settings`), not relative.
- **Datasets and model weights are not committed.** Keep them under `data/` (gitignored). Track only structure with `.gitkeep` / README files.
- **Notebooks** belong in `notebooks/` and should import from `pato` rather than redefining logic inline. Run `uv run jupyter lab` so the kernel uses the project venv.

## Git

The repo is initialised locally with **no remote** (`git remote -v` is empty). Do not add or push to a remote unless the user asks for it.
