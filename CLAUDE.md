# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

`patomorphology` — an AI pipeline that trains a model to predict cancer changes in **skin histopathology** images: H&E-stained tissue sections under a microscope (the "violet-and-pink" image type), typically as whole-slide images (WSIs) or large TIFF tiles. **Not** dermoscopy / clinical skin photographs — those are a separate modality and should not be assumed when the user says "images".

The Python package itself is named **`pato`** (Polish/European root for *patomorfologia* = histopathology) and lives under `src/pato/`.

### Domain primer (so future sessions don't re-derive this)

- **H&E staining:** hematoxylin → nuclei stain blue/violet; eosin → cytoplasm and stroma stain pink. This is what the user means by "violet cells under a microscope".
- **WSIs** are gigapixel images (commonly 100k × 100k px) at 20× or 40× magnification, stored in pyramidal formats (`.svs`, `.tiff`, `.ndpi`, `.mrxs`) — read with `openslide`/`tiatoolbox`/`cuCIM`, never loaded whole into memory.
- **Likely first dataset:** the Thomas et al. *Non-Melanoma Skin Cancer Segmentation* dataset (290 H&E tissue sections of BCC / SCC / IEC with **pixel-level masks** over 12 tissue classes, hosted on UQ Research Data Manager). Smaller, segmentation-ready, fits the "predict cancer changes" goal.
- **Other relevant sources** if the project scales: TCGA-SKCM (melanoma WSIs, slide-level labels only), TIL-WSI-TCGA, PanNuke (cross-tissue nucleus masks, includes skin).

## Toolchain

- **Package manager:** `uv` (pinned via `pyproject.toml` + `uv.lock`). Always use `uv` — never call `pip` or `python` directly.
- **Python:** 3.12 (pinned in `.python-version`).
- **Virtual environment:** `.venv/` in the repo root, managed by `uv sync`. Do not create or manage venvs by hand.
- **Models:** Use **Pydantic v2** (`BaseModel`) for all data classes / DTOs / configuration. Do **not** use `dataclasses.dataclass`. Application/runtime configuration uses `pydantic-settings` (`BaseSettings`).
- **ML stack:** PyTorch (`torch`) + Lightning (`lightning`, imported as `import lightning as L`). On macOS we run on Apple Silicon MPS — pick the device via `torch.device("mps" if torch.backends.mps.is_available() else "cpu")` (or let Lightning's `Trainer(accelerator="auto")` handle it).
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
