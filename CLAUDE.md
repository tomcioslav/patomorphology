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
- **Logged metrics:** every LightningModule logs `train_loss` (DiceCELoss) and, on validation, both `val_loss` and `val_dice` (`monai.metrics.DiceMetric`, mean across all classes). `val_loss` drives `ModelCheckpoint`'s best-checkpoint selection; `val_dice` is the segmentation-quality readout you compare across runs in **Weights & Biases** (`wandb.ai`).
- **Experiment tracking:** `wandb` via `lightning.pytorch.loggers.WandbLogger`. Project = `"patomorphology"`, run name = the auto-generated `run_name`, full resolved Hydra config logged as the W&B `config`. Requires a one-time `uv run wandb login`; set `WANDB_MODE=offline` to record locally without syncing.
- **Dataset boundary:** `pato.dataset.DatasetViewer` is the only public dataset class — and **inspection-only**: plain pydantic, `viewer[i]` returns a `PatoImage`. Used in notebooks like `explore_images.ipynb` and as the **input feed** for pipeline-specific torch Datasets: UNet's `UNetDataset` in `pipelines/unet/data.py`, and the `sam` pipeline's `SAMFeatureDataset` (frozen) and `SAMTileDataset` (end-to-end) in `pipelines/sam/data.py`. Each pipeline defines whatever torch Datasets it needs in its own `data.py` — they don't go under `pato.dataset` because what counts as "one training sample" differs per pipeline (tile vs SAM-feature vs whatever comes next). The pipeline torch Datasets all return tensor pairs directly, so default collation works everywhere — no `collate_fn` plumbing.
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
uv run wandb login                       # one-time per machine: paste API key from wandb.ai/authorize
                                         # then every train run streams metrics to
                                         # wandb.ai/<entity>/patomorphology

# Hydra-driven training. Four config groups compose every run:
#   net       — architecture (unet, unet_wide, sam, sam_deep, ...)
#   dataset   — which cache (nmsc-2x-unet-512, nmsc-2x-sam-vit-base-1024, ...)
#   lr        — learning rate value + scheduler (constant_1e4, cosine, step, ...)
#   pipeline  — training-loop shape (unet, sam [frozen], sam_finetune [end-to-end])
uv run python scripts/train.py                                            # defaults (UNet)
uv run python scripts/train.py net=unet_wide lr=cosine                    # different arch + LR schedule
uv run python scripts/train.py pipeline=sam net=sam_deep \
    dataset=nmsc-2x-sam-vit-base-1024 lr=constant_3e4                      # frozen SAM, head-only
uv run python scripts/train.py pipeline=sam_finetune net=sam \
    dataset=nmsc-2x-sam-full-1024 lr=constant_1e4 \
    init_from_checkpoint="'runs/<head-run>/checkpoints/best-....ckpt'"     # unfreeze + warm-start
uv run python scripts/train.py -m net=unet,unet_wide,unet_narrow          # 3-run net sweep
uv run python scripts/train.py -m \
    net=unet,unet_wide lr=constant_1e4,cosine                             # 4-run net × lr sweep
```

### Full dataset lifecycle

```python
from config import paths
from pato.dataset import DatasetViewer
from pato.dataset.builders import NMSCBuilder, SAMFeatureBuilder, TileBuilder

# 1. ONE-TIME per raw dataset: raw → normalized full-image dataset
NMSCBuilder(raw_root=paths.nmsc_5x).build(paths.data_processed / "nmsc-5x")
# data/processed/nmsc-5x/{metadata.json, samples/<id>.npz}  with canonical splits

# 2. ONE-TIME per pipeline-cache combination. Pick the right builder:
src = DatasetViewer(root=paths.data_processed / "nmsc-2x")

# UNet — raw 512 tiles
TileBuilder(source=src, target_size=512).build(paths.data_processed / "nmsc-2x-unet-512")
# sam_finetune (end-to-end) — raw 1024 tiles (same builder, different size)
TileBuilder(source=src, target_size=1024).build(paths.data_processed / "nmsc-2x-sam-full-1024")
# sam (frozen) — SAM-encoded 1024 tiles (pre-encoded; only useful with a frozen encoder)
SAMFeatureBuilder(source=src).build(paths.data_processed / "nmsc-2x-sam-vit-base-1024")

# Every cache records `source_root` so the run-analysis notebook can trace
# back to full images regardless of which cache the run trained on.

# 3. Train. Hydra composes the config from net + dataset + lr + pipeline:
#   uv run python scripts/train.py pipeline=sam net=sam \
#       dataset=nmsc-2x-sam-vit-base-1024 lr=constant_1e3
```

All builders are idempotent — re-running skips files that already exist on disk.

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
├── runs/                    # one folder per training run (Hydra's per-job dir): config.yaml +
│                            # checkpoints/ + wandb/ + .hydra/ + train.log (gitignored except README)
├── data/
│   ├── raw/                 # original datasets, untouched (e.g. nmsc-segmentation/)
│   └── processed/           # **canonical** form pipelines train on: one dir per (source × resolution),
│                            # plus per-pipeline derived caches (e.g. SAM features).
├── notebooks/               # explore_images.ipynb, explore_run.ipynb
├── conf/                    # Hydra configs — four composable groups:
│   ├── config.yaml          #   top-level: defaults + max_epochs + init_from_checkpoint + hydra block
│   ├── net/                 #   architecture (unet, unet_wide, sam, sam_deep, …)
│   ├── dataset/             #   which cache to train on (one-liner per cache)
│   ├── lr/                  #   learning rate value + scheduler (constant_*, cosine, step)
│   └── pipeline/            #   training-loop shape (unet, sam [frozen], sam_finetune [end-to-end])
├── scripts/                 # Hydra entry points (`train.py` decorated with `@hydra.main`)
├── config.py                # project-level pydantic `Paths` (root, data dirs, dataset version dirs, runs)
├── src/pato/                # main package
│   ├── __init__.py
│   ├── components/          # Reusable building blocks instantiated by Hydra `_target_`:
│   │   └── models/          #   - unet.py             `UNet` (wraps monai.networks.nets.UNet)
│   │                        #   - sam_head.py         `SAMSegHead`
│   │                        #   - sam_segmentation.py `SAMEncoder` (offline numpy encoder for
│   │                        #                          SAMFeatureBuilder), `SAMImageEncoder`
│   │                        #                          (differentiable nn.Module), and
│   │                        #                          `SAMSegmentation` (encoder + head,
│   │                        #                          the unified inference model)
│   ├── dataset/             # PROCESSED dataset format + builders that produce it:
│   │   ├── dataset_view.py  #   DatasetViewer — reads any data/processed/<name>/. Bare cache
│   │   │                    #     names (e.g. "nmsc-2x-unet-512") resolve to
│   │   │                    #     data/processed/<name> via a pydantic validator.
│   │   └── builders/        #   DatasetBuilder ABC + three concrete builders:
│   │                        #   - `nmsc.py` (NMSCBuilder)        raw → normalized full images
│   │                        #   - `tiles.py` (TileBuilder)       normalized → raw RGB tile cache
│   │                        #                                     (UNet at 512, sam_finetune at 1024)
│   │                        #   - `sam.py` (SAMFeatureBuilder)   normalized → SAM-feature cache
│   │                        #                                     (the frozen `sam` pipeline only)
│   ├── inference.py         # generic `Predictor` — sliding-window stitch + argmax
│   │                        #   over any image→logits `nn.Module`
│   ├── pipelines/           # Training-loop shapes. Each `__init__.py` exposes
│   │                        # `build(cfg, net) → (LightningModule, train_loader, val_loader)`
│   │                        # — composes injected net + LR partial + dataloader factory.
│   │   ├── train.py         #   pipeline-agnostic `train(cfg, runs_dir)` — instantiates net,
│   │   │                    #     imports `pato.pipelines.<cfg.pipeline.name>` via importlib,
│   │   │                    #     calls its `build(cfg, net)`, sets up Trainer. Also
│   │   │                    #     `warm_start_model` for `init_from_checkpoint`.
│   │   ├── unet/            #   data.py (PatoImage collate over TileBuilder cache);
│   │   │                    #     module.py (UNetLightning takes `model` via DI).
│   │   └── sam/             #   Unified SAM pipeline. `cfg.pipeline.sam_frozen` picks the
│   │                        #     regime: frozen → SAMFeatureDataset over the feature cache,
│   │                        #     head-only optimizer; unfrozen → DatasetViewer over the 1024
│   │                        #     tile cache, two-group optimizer. `self.model` is always a
│   │                        #     full `SAMSegmentation`, so checkpoints are structurally
│   │                        #     identical across regimes (warm-start = plain state_dict load).
│   ├── experiments.py       # `list_runs`, `load_run`, `load_predictor`, `load_inference_model`,
│   │                        #   `source_dataset_root` — notebook-facing helpers
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

No dataset-level extras (class names, class count, dataset name, etc.) — those live on the model (`UNet.num_classes`, etc.) or are derived from sample IDs at usage time.

Two roles for this same shape:

1. **Normalized dataset** (e.g. `data/processed/nmsc-2x/`) — produced by `NMSCBuilder`. The `image` array is RGB `(H, W, 3) uint8` (full slide). Read via `DatasetViewer` for inspection and as the source feed for the cache builders.
2. **Pipeline cache** — produced by `TileBuilder` (raw RGB tiles, used by UNet + `sam_finetune`) or `SAMFeatureBuilder` (SAM features, the frozen `sam` pipeline only). The `image` array shape depends on the builder. Splits are projected forward from the upstream normalized dataset (one source image → N tile IDs, all in the same split as their source). Every cache records `source_root` in `metadata.config` so notebooks can trace back to step (1).

Both forms are read with the same `DatasetMetadata` schema.

Pipelines select splits **by name** — never by random `torch.randperm`. Two runs pointing at the same `data/processed/<name>/` always use the identical partition.

## Conventions

- **Configuration:** the root-level `config.py` (not inside the `pato` package) holds project paths. Import as `from config import paths`. Override any field with `PATO_PATHS_`-prefixed env vars (e.g. `PATO_PATHS_NMSC_5X=/abs/path`). `pato`-internal modules accept paths as parameters rather than importing this config directly.
- **Hydra config groups:** every run composes four groups (`net`, `dataset`, `lr`, `pipeline`) plus top-level scalars (`max_epochs`, `fast_dev_run`). Each group is a dir under `conf/`; the chosen yaml is selected via `defaults:` in `conf/config.yaml` or on the CLI (`net=unet_wide`). The `net:` group has `_target_: pato.components.models.X` so Hydra `instantiate(cfg.net)` rebuilds the model; the `lr:` group's `scheduler` has `_partial_: true` so it can be applied to an optimizer at runtime.
- **Pipelines:** each pipeline lives in `src/pato/pipelines/<name>/` with a fixed shape:
  - `__init__.py` — exposes `build(cfg, net) → (LightningModule, train_loader, val_loader)`. The **only** symbol `pato.pipelines.train.train` calls into.
  - `module.py` — Lightning module. `__init__` takes `model: nn.Module` (injected), `learning_rate: float`, `scheduler_partial: Callable | None`, plus pipeline-specific extras (`sam_frozen` + `sam_learning_rate` for the `sam` pipeline). Saves hparams *except* `model` and `scheduler_partial` (not yaml-serializable). Exposes `to_inference_model() → nn.Module` for predictor reconstitution.
  - `data.py` — dataloader factory/factories and any pipeline-specific torch Datasets. The `sam` pipeline has two factories (`make_feature_dataloaders` / `make_tile_dataloaders`); `build()` picks based on `sam_frozen`.
- **SAM pipeline is unified:** one `sam` pipeline, one `SAMLightning`, controlled by `cfg.pipeline.sam_frozen`. `self.model` is *always* a full `SAMSegmentation` (encoder + head) — the flag only changes which params have `requires_grad`, which `forward` branch runs (cached features vs raw images), and the optimizer group structure. Because the module structure never changes, every checkpoint has identical `model.encoder.* + model.head.*` keys, so warm-starting an unfrozen run from a frozen one (`pipeline=sam_finetune init_from_checkpoint=…`) is a plain `state_dict` load. `conf/pipeline/sam.yaml` is the frozen preset; `conf/pipeline/sam_finetune.yaml` is the unfrozen preset (both `name: sam`).
- **Pipeline-agnostic training entry:** `pato.pipelines.train.train(cfg, runs_dir)` instantiates the net via Hydra, imports `pato.pipelines.<cfg.pipeline.name>` via `importlib`, calls its `build(cfg, net)`, sets up the Trainer. Adding a new pipeline = drop a package with `build()` in `__init__.py`; nothing in `train.py` changes.
- **Run directory:** Hydra's per-job directory **is** the run directory — `conf/config.yaml`'s `hydra.run.dir` / `hydra.sweep` block points it at `runs/<pipeline>-<net>-<dataset>-<timestamp>[-<job>]`. `scripts/train.py` reads `HydraConfig.get().runtime.output_dir` and passes it to `train(cfg, run_dir)`. So one folder holds both Hydra's bookkeeping (`.hydra/`, `train.log`) and the run artifacts (`config.yaml`, `checkpoints/`, `wandb/`). There is no separate `outputs/` tree.
- **Run config persistence:** training saves the resolved Hydra cfg as `runs/<run>/config.yaml`. `pato.experiments.load_inference_model(run_path)` rebuilds the net via `instantiate(cfg.net)`, then `LightningModule.load_from_checkpoint(ckpt, model=net).to_inference_model()`.
- **W&B:** `train()` closes its W&B run (`wandb.finish()`) in a `finally` after `trainer.fit()` — Hydra multirun (`-m`) runs every sweep job in one process, so without this the next job's `WandbLogger` silently reuses the previous run.
- **Run names** auto-generated as `{pipeline}-{source}-{source_root.name}-{timestamp}` unless overridden.
- **Data classes:** every structured object that crosses a module boundary is a `pydantic.BaseModel`. No `dataclasses`, no `TypedDict` for data.
- **Imports:** absolute (`from pato.source import NMSCDataset`), not relative.
- **Datasets, processed caches, and model weights are not committed.** Track only structure with `.gitkeep` / README files.
- **Notebooks** import from `pato` rather than redefining logic inline. Run `uv run jupyter lab` so the kernel uses the project venv.

## Git

The repo is initialised locally with **no remote** (`git remote -v` is empty). Do not add or push to a remote unless the user asks for it.
