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
- **ML stack:** PyTorch (`torch`) + Lightning (`lightning`, imported as `import lightning as L`) + **MONAI** (`monai`). On macOS we run on Apple Silicon MPS — pick the device via `torch.device("mps" if torch.backends.mps.is_available() else "cpu")` (or let Lightning's `Trainer(accelerator="auto")` handle it).
- **MONAI usage policy:** prefer MONAI for things that are domain-standard rather than re-implementing them. Specifically use:
  - `monai.inferers.sliding_window_inference` for tile-and-stitch inference (Gaussian-weighted blending, batched, GPU-friendly — never write our own join).
  - `monai.networks.nets.UNet` / `SegResNet` / `SwinUNETR` for model architectures.
  - `monai.losses.DiceCELoss` (or `DiceLoss`, `FocalLoss`) for segmentation losses.
  - `monai.metrics.DiceMetric`, `MeanIoU` for evaluation.
  - `monai.transforms.RandSpatialCropd` / `RandCropByPosNegLabeld` for training-time random crops (only when training end-to-end — irrelevant for the SAM-as-frozen-backbone pipeline, which uses pre-computed deterministic tiles).
  Keep what's project-specific in `pato`: `PatoImage`, `BaseImageMaskDataset`/`NMSCDataset`, `split_image` (deterministic tiling for cached-feature pipelines), `pato.visualize`, `config.py`.
- **Tensor convention:** `PatoImage.to_torch()` returns `(image, mask)` where image is `(3, H, W)` `float32` in [0, 1] and mask is `(H, W)` `int64` class indices. This is what `monai` losses and `sliding_window_inference` expect; add a leading batch dim with `.unsqueeze(0)` at the call site.
- **Dataset hierarchy (single paradigm):** every dataset in `pato.source` is **both** a `pydantic.BaseModel` (or plain class) **and** a `torch.utils.data.Dataset`, and `dataset[i]` always returns a `PatoImage` — never raw tensors. To use with `DataLoader`, pass `collate_fn=PatoImage.collate`, which stacks a list of `PatoImage` into batched `(B, 3, H, W)` / `(B, H, W)` tensors. This keeps the loader / tiler / pipeline boundary clean: only the collate step crosses into tensor land.
- **Visualization:** **plotly** (not matplotlib). Image / mask viewers live in `pato.visualize`. Image loading uses Pillow with `Image.MAX_IMAGE_PIXELS = None` set in the module (the dataset's native-resolution TIFFs exceed Pillow's default decompression-bomb cap).
- **Build backend:** `uv_build` (declared in `pyproject.toml`). Source layout is `src/`.

## Common commands

```bash
uv sync                                  # install / refresh deps and the venv
uv run nbstripout --install \
    --attributes .gitattributes          # one-time per clone: install the
                                         # nbstripout git filter
uv add <pkg>                             # add a runtime dependency
uv add --group dev <pkg>                 # add a dev dependency
uv run python -m <module>                # run code inside the project venv
uv run pytest                            # run tests
uv run ruff check .                      # lint
uv run ruff format .                     # format
uv run jupyter lab                       # start Jupyter for notebooks/
uv run python -m pato.pipelines.unet.train    # train the U-Net (writes to runs/<run_name>/)
uv run tensorboard --logdir runs              # live loss curves at localhost:6006 (all runs)
```

## Notebooks

`*.ipynb` files are filtered through **nbstripout** on commit (configured in
the tracked `.gitattributes`). This keeps notebook diffs to source changes only —
outputs, execution counts, and per-cell metadata never enter git history. The
filter only activates after a fresh clone runs `uv run nbstripout --install`,
so include that step in any onboarding instructions.

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
├── runs/                    # one folder per training run: config.yaml + checkpoints/ + tensorboard/ (gitignored except README)
├── data/
│   ├── raw/                 # original datasets, untouched
│   └── processed/           # cached/derived per (source × pipeline-preprocess) — only when a pipeline needs caching (e.g. SAM features)
├── notebooks/               # explore_images.ipynb (sources), explore_runs.ipynb (load any run + predict)
├── config.py                # **project-level** pydantic `Paths` (root, data dirs, dataset version dirs, runs)
├── src/pato/                # main package
│   ├── __init__.py
│   ├── source/              # raw on-disk readers — `BaseImageMaskDataset`, `NMSCDataset`, `TiledDataset`
│   ├── pipelines/
│   │   ├── base.py          # `BasePredictor` ABC
│   │   └── unet/            # one self-contained pipeline: `config.py` (RunConfig), `module.py`, `data.py`, `train.py`, `predictor.py`
│   ├── experiments.py       # `list_runs`, `load_run`, `load_predictor` — notebook-facing helpers
│   ├── models/              # `build_unet` factory around `monai.networks.nets.UNet`
│   ├── schema/              # `PatoImage` pydantic model (paired image + mask)
│   ├── utils/               # `split_image` — overlapping-tile generator
│   └── visualize/           # plotly-based image / mask viewers
└── tests/                   # pytest tests
```

`data/raw/`, `data/processed/`, `runs/`, and `tests/` track only structure (READMEs / `.gitkeep`); their contents are gitignored.

## Conventions

- **Configuration:** the root-level `config.py` (not inside the `pato` package) holds project paths. Import as `from config import paths`. Override any field with `PATO_PATHS_`-prefixed env vars (e.g. `PATO_PATHS_NMSC_5X=/abs/path`). `pato`-internal modules accept paths as parameters rather than importing this config directly.
- **Pipelines:** each pipeline lives in `src/pato/pipelines/<name>/` with a fixed file shape:
  - `config.py` — `<Pipeline>RunConfig` (pydantic) — fully describes a run.
  - `module.py` — Lightning module (training-side).
  - `data.py` — `make_dataloaders(config)` (training-side).
  - `train.py` — entry point: `train(config, runs_dir, run_name=None)` writes everything to `runs/<run_name>/`.
  - `predictor.py` — `<Pipeline>Predictor(BasePredictor)` (inference-side mirror). `from_run(run)` rehydrates from a saved run; `predict(image)` accepts `Path | str | np.ndarray | PatoImage` and returns `(H, W) int`.
  - (Pipelines that need caching, e.g. SAM-head: also `preprocess.py` writing to `data/processed/<name>/` and a cache reader, both colocated here — never in a shared `pato/cached/` namespace.)
- **Run tracking:** every `train(config, runs_dir)` writes one folder under `runs/<run_name>/` with `config.yaml`, `checkpoints/`, `tensorboard/`. Notebooks load any run via `pato.experiments.load_predictor(run_path)` — dispatches on `config.pipeline` to the right `Predictor` subclass.
- **Run names** auto-generated as `{pipeline}-{source}-{source_root.name}-{timestamp}` unless overridden.
- **Data classes:** every structured object that crosses a module boundary is a `pydantic.BaseModel`. No `dataclasses`, no `TypedDict` for data.
- **Imports:** absolute (`from pato.source import NMSCDataset`), not relative.
- **Datasets, processed caches, and model weights are not committed.** Track only structure with `.gitkeep` / README files.
- **Notebooks** import from `pato` rather than redefining logic inline. Run `uv run jupyter lab` so the kernel uses the project venv.

## Git

The repo is initialised locally with **no remote** (`git remote -v` is empty). Do not add or push to a remote unless the user asks for it.
