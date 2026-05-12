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
- **Logged metrics:** every LightningModule logs `train_loss` (DiceCELoss) and, on validation, both `val_loss` and `val_dice` (`monai.metrics.DiceMetric`, mean across all classes). `val_loss` drives `ModelCheckpoint`'s best-checkpoint selection; `val_dice` is the segmentation-quality readout you compare across runs in TensorBoard.
- **Dataset boundary:** `pato.dataset.DatasetViewer` is the only public dataset class. `viewer[i]` returns a `PatoImage`. It's used for two things: **inspection** (notebooks like `explore_images.ipynb`) and as the **input feed** for pipeline-specific torch Datasets (UNet's `UNetDataset` in `pipelines/unet/data.py`, SAM-head's `SAMFeatureDataset` in `pipelines/sam_head/data.py`). Each pipeline defines whatever torch Datasets it needs in its own `data.py` — they don't go under `pato.dataset` because what counts as "one training sample" differs per pipeline (tile vs SAM-feature vs whatever comes next). For `DataLoader` batching of `PatoImage`-emitting Datasets, pass `collate_fn=PatoImage.collate`.
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
uv run tensorboard --logdir runs              # live loss curves at localhost:6006 (all runs)

# Hydra-driven training (single run + multirun sweeps)
uv run python scripts/train.py                                            # default UNet
uv run python scripts/train.py pipeline=unet_quick                        # smaller variant
uv run python scripts/train.py pipeline=sam_head                          # SAM-head pipeline
uv run python scripts/train.py -m pipeline=unet,unet_quick                # two runs
uv run python scripts/train.py -m \
    pipeline.target_size=256,512 pipeline.learning_rate=1e-3,1e-4         # 4-run sweep
```

### Full dataset lifecycle

```python
from config import paths
from pato.dataset import DatasetViewer
from pato.dataset.convert.nmsc import convert
from pato.pipelines.sam_head.preprocess import preprocess
from pato.pipelines.sam_head.config import SAMHeadRunConfig
from pato.pipelines.train import train

# 1. ONE-TIME per raw dataset: convert raw → normalized
convert(raw_root=paths.nmsc_5x, out_dir=paths.data_processed / "nmsc-5x")
# data/processed/nmsc-5x/{metadata.json, samples/<id>.npz}  with canonical splits

# 2. ONE-TIME per pipeline-with-preprocessing: build the SAM cache
src = DatasetViewer(root=paths.data_processed / "nmsc-2x")
cache_dir = paths.data_processed / "nmsc-5x-sam-vit-base-1024"
preprocess(src, cache_dir=cache_dir)
# splits in src.metadata are inherited into the cache's manifest

# 3. Train the head — fast (head is ~175k params; SAM is frozen)
cfg = SAMHeadRunConfig(
    dataset_root=paths.data_processed / "nmsc-5x",
    cache_dir=cache_dir,
)
train(cfg, runs_dir=paths.runs)
```

Both `convert` and `preprocess` are idempotent — re-running skips files
that already exist on disk.

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
│   ├── raw/                 # original datasets, untouched (e.g. nmsc-segmentation/)
│   └── processed/           # **canonical** form pipelines train on: one dir per (source × resolution),
│                            # plus per-pipeline derived caches (e.g. SAM features).
├── notebooks/               # explore_images.ipynb, explore_run.ipynb
├── conf/                    # Hydra configs (top-level `config.yaml` + `pipeline/*.yaml` variants)
├── scripts/                 # Hydra entry points (`train.py` decorated with `@hydra.main`)
├── config.py                # project-level pydantic `Paths` (root, data dirs, dataset version dirs, runs)
├── src/pato/                # main package
│   ├── __init__.py
│   ├── dataset/             # PROCESSED dataset format and per-source converters:
│   │                        #   `dataset_view.py` (DatasetViewer — reads data/processed/<name>/, used for
│   │                        #     **inspection** and as the source feed for pipeline-specific Datasets),
│   │                        #   `convert/<source>.py` (one converter per raw source, e.g. `convert/nmsc.py`).
│   │                        # Pipeline-specific torch Datasets (tiled, feature-cached, etc.) live INSIDE
│   │                        # the pipeline that owns them — not here.
│   ├── pipelines/
│   │   ├── base.py          # `BaseRunConfig` + `BasePredictor` ABC
│   │   ├── train.py         # pipeline-agnostic `train(config, runs_dir, …)` — uses `importlib` to call `pato.pipelines.{config.pipeline}.build(config)`; never imports a specific pipeline
│   │   ├── unet/            # end-to-end UNet: `config.py`, `model.py` (architecture),
│   │   │                    #   `module.py` (LightningModule), `data.py`, `predictor.py`
│   │   ├── sam_head/        # SAM-as-frozen-backbone + trainable head:
│   │   │                    #   `model.py` (frozen `SAMEncoder` + trainable `SAMSegHead` —
│   │   │                    #     shared by preprocess + predictor; one source of truth),
│   │   │                    #   `module.py` (LightningModule wrapping SAMSegHead),
│   │   │                    #   `preprocess.py` (DatasetViewer → data/processed/<name>/.npz cache),
│   │   │                    #   `data.py` (SAMFeatureDataset reads the cache + make_dataloaders),
│   │   │                    #   `predictor.py`
│   │   └── sam_full/        # End-to-end SAM encoder + head, both trainable. No
│   │                        # feature cache — `data.py` tiles raw images on the fly
│   │                        # because backprop has to flow through SAM. Two
│   │                        # optimizer groups (SAM at small LR, head at larger LR).
│   ├── experiments.py       # `list_runs`, `load_run`, `load_predictor` — notebook-facing helpers
│   ├── schema/              # `PatoImage`, `DatasetMetadata`, `SampleMetadata` (pydantic)
│   ├── utils/               # `split_image` — overlapping-tile generator
│   └── visualize/           # plotly-based image / mask viewers
└── tests/                   # pytest tests
```

`data/raw/`, `data/processed/`, `runs/`, and `tests/` track only structure (READMEs / `.gitkeep`); their contents are gitignored.

## Dataset lifecycle

Everything under `data/processed/<name>/` follows **the same shape**, regardless of whether it's a raw-source conversion or a pipeline-specific cache:

```
data/processed/<name>/
├── metadata.json              # DatasetMetadata: just `splits` + `samples`
└── samples/<id>.npz           # {image: (...) array fed to the net, mask: (H, W) uint8 class IDs}
```

`DatasetMetadata` is intentionally minimal:
- `splits: dict[str, list[str]]` — split name → list of sample IDs
- `samples: dict[str, SampleMetadata]` — sample ID → `{path, size}`

No dataset-level extras (class names, class count, dataset name, etc.) — those either live in pipeline `RunConfig`s or are derived from sample IDs at usage time.

Two roles for this same shape:

1. **Normalized dataset** (e.g. `data/processed/nmsc-5x/`) — produced once per raw source by a converter in `pato.dataset.<source>`. The `image` array is RGB `(H, W, 3) uint8`. Read via `DatasetViewer` for inspection and as the input feed for pipeline-specific Datasets.
2. **Pipeline-specific cache** (e.g. `data/processed/nmsc-5x-sam-vit-base-1024/`) — produced by a pipeline's `preprocess.py` when the pipeline benefits from caching an expensive transform (SAM encoding). The `image` array is whatever-feeds-the-net for that pipeline (e.g. `(256, 64, 64) float32` SAM features). Splits are projected forward from the upstream normalized dataset (one source image → N tile IDs, all in the same split as their source).

Both forms are read with the same `DatasetMetadata` schema. Pipeline-specific Datasets (in each pipeline's `data.py` / `cache.py`) wrap the appropriate cache and produce whatever batched tensors the model expects.

Pipelines select splits **by name** — never by random `torch.randperm`. Two runs pointing at the same `data/processed/<name>/` always use the identical partition.

## Conventions

- **Configuration:** the root-level `config.py` (not inside the `pato` package) holds project paths. Import as `from config import paths`. Override any field with `PATO_PATHS_`-prefixed env vars (e.g. `PATO_PATHS_NMSC_5X=/abs/path`). `pato`-internal modules accept paths as parameters rather than importing this config directly.
- **Pipelines:** each pipeline lives in `src/pato/pipelines/<name>/` with a fixed file shape:
  - `__init__.py` — exports `build(config) → (LightningModule, train_loader, val_loader)`. This is the **only** symbol `pato.pipelines.train.train` calls into; everything else is internal to the pipeline.
  - `config.py` — `<Pipeline>RunConfig` (pydantic) — subclass of `BaseRunConfig`. Adds a `pipeline: Literal["<name>"]` discriminator that **must match the package directory name** (so `importlib.import_module(f"pato.pipelines.{pipeline}")` resolves).
  - `model.py` — model architecture (plain `torch.nn.Module` or factory).
  - `module.py` — Lightning module (training-side); imports `model.py`.
  - `data.py` — torch `Dataset`(s) the pipeline needs (`UNetDataset`, `SAMFeatureDataset`, etc.) **and** `make_dataloaders(config)`. One file per pipeline.
  - `predictor.py` — `<Pipeline>Predictor(BasePredictor)`; `from_run(run)` rehydrates, `predict(image)` returns `(H, W) int`.
  - (Pipelines that need caching, e.g. SAM-head: also `preprocess.py` writing to `data/processed/<name>/`. Any frozen backbone shared between preprocess and predictor lives in `model.py` alongside the trainable head — same file, different roles.)
- **Pipeline-agnostic training entry:** `pato.pipelines.train.train(config, runs_dir, …)` reads only `config.pipeline` / `dataset_root` / `max_epochs` (the fields declared on `BaseRunConfig`), then `importlib.import_module(f"pato.pipelines.{config.pipeline}").build(config)` to get the model + loaders. **`train.py` imports zero specific pipelines.** Adding a new pipeline = drop in a new package with the standard files + `build` in `__init__.py`; nothing in `train.py` changes. This is also the seam for moving to Hydra later — swap the `importlib` hop for `hydra.utils.instantiate` on a `_target_` field and the rest of the file is unaffected.
- **Run tracking:** every `train(config, runs_dir)` writes one folder under `runs/<run_name>/` with `config.yaml`, `checkpoints/`, `tensorboard/`. Notebooks load any run via `pato.experiments.load_predictor(run_path)` — dispatches on `config.pipeline` to the right `Predictor` subclass.
- **Run names** auto-generated as `{pipeline}-{source}-{source_root.name}-{timestamp}` unless overridden.
- **Data classes:** every structured object that crosses a module boundary is a `pydantic.BaseModel`. No `dataclasses`, no `TypedDict` for data.
- **Imports:** absolute (`from pato.source import NMSCDataset`), not relative.
- **Datasets, processed caches, and model weights are not committed.** Track only structure with `.gitkeep` / README files.
- **Notebooks** import from `pato` rather than redefining logic inline. Run `uv run jupyter lab` so the kernel uses the project venv.

## Git

The repo is initialised locally with **no remote** (`git remote -v` is empty). Do not add or push to a remote unless the user asks for it.
