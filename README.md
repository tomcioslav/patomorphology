# patomorphology

Training pipelines for segmenting cancer changes in **skin histopathology** (H&E-stained tissue under a microscope — the "violet-and-pink" image type).

> **Scope.** This repo is **model-building only**: dataset preprocessing, training, evaluation, and run analysis in notebooks. No serving, no inference API, no clinical UI. Trained checkpoints live in `runs/` and are consumed by notebooks or downstream projects.

The Python package is **`pato`** (from Polish *patomorfologia*) under `src/pato/`.

---

## Setup

```bash
uv sync                                  # install deps + create .venv/
uv run nbstripout --install \
    --attributes .gitattributes          # one-time: strip notebook outputs on commit
uv run wandb login                       # one-time: paste API key from wandb.ai/authorize
```

Everything runs through `uv` — never call `pip` or `python` directly.

---

## Data

Everything under `data/processed/` follows **the same on-disk shape**, regardless of whether it's a normalized full-image dataset or a pipeline-specific cache:

```
data/processed/<name>/
├── metadata.json              # DatasetMetadata: splits + samples
└── samples/<id>.npz           # {image: <array fed to the net>, mask: (H, W) uint8 class IDs}
```

`DatasetMetadata` is intentionally minimal — just `splits: dict[str, list[str]]` (split name → sample IDs) and `samples: dict[str, SampleMetadata]` (sample ID → `{path, size}`). No dataset-level extras like class names or class count — those belong on the model.

### One canonical dataset, several pipeline caches

```
data/raw/nmsc-segmentation/         # original UQ dataset, untouched
data/processed/
├── nmsc-2x/                        # CANONICAL — normalized full-image dataset
├── nmsc-2x-unet-256/               #  cache: 256-px raw RGB tiles
├── nmsc-2x-unet-512/               #  cache: 512-px raw RGB tiles
├── nmsc-2x-sam-full-1024/          #  cache: 1024-px raw RGB tiles (SAM end-to-end)
└── nmsc-2x-sam-vit-base-1024/      #  cache: pre-encoded SAM features
```

Two roles for the same on-disk shape:

1. **`nmsc-2x/` — the normalized full-image dataset.** Produced once by `NMSCBuilder` from `data/raw/nmsc-segmentation/`. Stores full slides as RGB `(H, W, 3) uint8` with class-id masks `(H, W) uint8`. Records canonical train/val/test splits. **Every pipeline validates against this dataset** via sliding-window inference, which is why `val_dice` is directly comparable across runs no matter what the model trained on.

2. **`nmsc-2x-<pipeline>-<size>/` — pipeline-specific training caches.** Built once per (model family, input size) combination from the canonical dataset:
   - `TileBuilder` cuts the full slides into raw RGB tiles → `nmsc-2x-unet-512` (UNet), `nmsc-2x-sam-full-1024` (SAM end-to-end).
   - `SAMFeatureBuilder` runs each tile through the frozen SAM ViT once and saves the `(256, 64, 64)` feature volume → `nmsc-2x-sam-vit-base-1024`. Training the head on cached features is ~100× faster than re-encoding every step.

   Splits are projected from the canonical dataset (one source image → N tile IDs, all in the same split as their source). Each cache records `source_root` in its metadata so we can trace back to the canonical full images at any time.

**Rule of thumb:** training reads from a cache (fast, fixed tile size). Validation always reads from `nmsc-2x` (slow but apples-to-apples across pipelines).

### Why every pipeline validates against the same `nmsc-2x`

Training tile size is an engineering choice — a UNet trains on 512-px tiles because that's what fits a sensible batch, a SAM head trains on 1024-px tiles because that's the encoder's native size. If we computed Dice over those training tiles, the score would be entangled with the tile size: smaller tiles have less context, edges of tiles bias the score, and a run on 256-px tiles would be measuring something subtly different than a run on 1024-px tiles. Comparing `val_dice` across pipelines would be meaningless.

The fix is to make validation independent of training tile size:

1. **Every pipeline validates against the canonical `nmsc-2x`** — full slides, not tiles. This is enforced in the config: every `conf/dataset/*.yaml` pins `val: nmsc-2x`.
2. **The model predicts the whole image** via `monai.inferers.sliding_window_inference` — tile, predict, Gaussian-weighted stitch back to full image size. The tile size used here is read from the **training cache's** metadata, so it matches what the model was trained on. There's no per-pipeline val math; the shared logic lives in [`pato.pipelines._val`](src/pato/pipelines/_val.py).
3. **`val_dice` is computed once, on the full-image prediction** — `monai.metrics.DiceMetric`, mean across classes, one number per slide. No per-tile averaging.

Net effect: `val_dice` is the same protocol for every run. UNet @ 512 vs SAM @ 1024 vs whatever comes next — all directly comparable, all selecting checkpoints by the same yardstick. The training tile size affects training speed and memory, not the validation metric.

There is **no `val_loss`** — full-image Dice is the single val-side signal.

All builders are idempotent — re-running skips existing files. A full lifecycle from raw → canonical → caches:

```python
from config import paths
from pato.dataset import DatasetViewer
from pato.dataset.builders import NMSCBuilder, SAMFeatureBuilder, TileBuilder

# 1. ONE-TIME per raw dataset: raw → normalized full-image dataset
NMSCBuilder(raw_root=paths.nmsc_5x).build(paths.data_processed / "nmsc-5x")

# 2. ONE-TIME per pipeline cache:
src = DatasetViewer(root=paths.data_processed / "nmsc-2x")
TileBuilder(source=src, target_size=512).build(paths.data_processed / "nmsc-2x-unet-512")
TileBuilder(source=src, target_size=1024).build(paths.data_processed / "nmsc-2x-sam-full-1024")
SAMFeatureBuilder(source=src).build(paths.data_processed / "nmsc-2x-sam-vit-base-1024")
```

### Visualization

[`notebooks/explore_images.ipynb`](notebooks/explore_images.ipynb) walks through any processed dataset — it lists splits, samples, and shows image/mask overlays via the plotly viewers in [`pato.visualize`](src/pato/visualize/). Open it whenever you want to spot-check a freshly built cache.

---

## Models

Two architectures live under [`src/pato/components/models/`](src/pato/components/models/) — both are plain `nn.Module`s instantiated by Hydra (`_target_: ...` in `conf/net/*.yaml`):

### UNet — [unet.py](src/pato/components/models/unet.py)

Thin wrapper around `monai.networks.nets.UNet`. Parameterized by `channels` (depth + width) and `num_res_units`. Variants — `unet`, `unet_wide`, `unet_narrow`, `unet_shallow` — differ only in `channels`/`num_res_units` in `conf/net/unet*.yaml`.

### SAM — [sam_segmentation.py](src/pato/components/models/sam_segmentation.py) + [sam_head.py](src/pato/components/models/sam_head.py)

Composed of two pieces:

- **`SAMImageEncoder`** — Meta's SAM ViT-B (`facebook/sam-vit-base`), differentiable `nn.Module`. Output: `(256, 64, 64)` features regardless of SAM size.
- **`SAMSegHead`** — an upsampling decoder (four 2× transposed-conv stages) that turns `(256, 64, 64)` features into a `1024×1024` segmentation map. Configurable widths and optional extra Conv3×3 blocks per stage.

`SAMSegmentation` glues encoder + head together. It's used in **both** training regimes (frozen and end-to-end) so checkpoints have an identical key structure — warm-starting an end-to-end run from a frozen head-only checkpoint is a plain `state_dict` load. Net variants — `sam`, `sam_deep`, `sam_wide`, `sam_narrow` — differ in head capacity (`widths`, `blocks_per_stage`).

A separate `SAMEncoder` class exists in the same file but is **offline-only** (numpy in/out, `@torch.no_grad()`) — it's the encoder `SAMFeatureBuilder` uses to populate the SAM feature cache. It's not part of any training graph.

### Repo layout for the model code

```
src/pato/
├── components/models/      # UNet, SAMImageEncoder, SAMSegHead, SAMSegmentation
├── dataset/
│   ├── dataset_view.py     # DatasetViewer — reads any data/processed/<name>/
│   └── builders/           # NMSCBuilder, TileBuilder, SAMFeatureBuilder
├── pipelines/              # training-loop shapes (see next section)
│   ├── unet/               # UNetLightning + UNetDataset
│   ├── sam/                # SAMLightning + SAMFeatureDataset / SAMTileDataset
│   ├── _val.py             # shared full-image sliding-window validation
│   └── train.py            # pipeline-agnostic train(cfg, run_dir)
├── inference.py            # generic sliding-window Predictor
├── experiments.py          # list_runs / load_run / load_inference_model
├── schema/                 # PatoImage, DatasetMetadata, SampleMetadata (pydantic v2)
├── utils/                  # split_image (overlapping-tile generator)
└── visualize/              # plotly image/mask viewers
```

A **pipeline** (under `src/pato/pipelines/<name>/`) is the training-loop shape: which LightningModule, which DataLoader factory, which optimizer setup. Three are implemented today:

| Pipeline       | Trains on                          | What runs                                    | When to use                          |
| -------------- | ---------------------------------- | -------------------------------------------- | ------------------------------------ |
| `unet`         | `nmsc-2x-unet-{256,512}`           | MONAI UNet end-to-end on RGB tiles           | Baseline / fastest iteration         |
| `sam`          | `nmsc-2x-sam-vit-base-1024`        | Frozen SAM-ViT encoder + trainable head      | Cheap head-only training on features |
| `sam_finetune` | `nmsc-2x-sam-full-1024`            | Full SAM-ViT + head, end-to-end fine-tune    | Squeeze more out after head pretrain |

`sam` and `sam_finetune` share the same code (`name: sam`) — `cfg.pipeline.sam_frozen` flips which `forward` branch runs and which parameters have `requires_grad`.

Adding a new pipeline = drop a package with `build(cfg, net) → (LightningModule, train_loader, val_loader)` in its `__init__.py`. `pato.pipelines.train.train` discovers it via `importlib`.

---

## Hydra

Every run is composed by **Hydra** from four config groups under `conf/`. Each group is a folder of yaml files; the chosen file is picked either by `defaults:` in `conf/config.yaml` or on the CLI.

```
conf/
├── config.yaml         # defaults + max_epochs + init_from_checkpoint + run-dir layout
├── net/                # architecture       (unet, unet_wide, sam, sam_deep, ...)
├── dataset/            # train cache + val source (nmsc-2x-unet-512, nmsc-2x-sam-full-1024, ...)
├── lr/                 # learning rate + scheduler (constant_1e4, cosine, step, cyclic)
└── pipeline/           # training-loop shape (unet, sam, sam_finetune)
```

What each group does:

- **`net/`** — pure architecture. `_target_: pato.components.models.X` so `hydra.utils.instantiate(cfg.net)` rebuilds the model. Number of classes, channel widths, depth, head config all live here.
- **`dataset/`** — pairs a **`train`** cache (the one the model trains on, e.g. `nmsc-2x-unet-512`) with a **`val`** source (always `nmsc-2x`). Every dataset yaml pins `val: nmsc-2x` so sliding-window validation runs the same protocol across all pipelines.
- **`lr/`** — learning rate value + scheduler. The scheduler is declared with `_partial_: true` so it can be applied to an optimizer at runtime.
- **`pipeline/`** — training-loop shape (`name: unet | sam`) plus pipeline-specific scalars: `batch_size`, `num_workers`, `precision`, and for SAM `sam_frozen`, `sam_learning_rate`, `gradient_checkpointing`.

Override any group on the CLI: `net=unet_wide lr=cosine dataset=nmsc-2x-unet-256`. Hydra multirun (`-m`) sweeps the cross-product.

Hydra's per-job directory **is** the run directory — `conf/config.yaml` points it at `runs/<pipeline>-<net>-<dataset>-<timestamp>/`. The training script reads `HydraConfig.get().runtime.output_dir` and writes `config.yaml` (resolved cfg) + `checkpoints/` + `wandb/` into the same folder, alongside Hydra's own `.hydra/` + `train.log`. No separate `outputs/` tree.

---

## Training

Default run (UNet on 512-px tiles, constant LR 1e-4, 15 epochs):

```bash
uv run python scripts/train.py
```

Frozen SAM head with cosine LR schedule for 30 epochs:

```bash
uv run python scripts/train.py \
    pipeline=sam \
    net=sam \
    dataset=nmsc-2x-sam-vit-base-1024 \
    lr=cosine \
    max_epochs=30
```

End-to-end SAM fine-tune, warm-started from the head-only run above:

```bash
uv run python scripts/train.py \
    pipeline=sam_finetune \
    net=sam \
    dataset=nmsc-2x-sam-full-1024 \
    lr=constant_1e4 \
    init_from_checkpoint="'runs/sam-sam-nmsc-2x-sam-vit-base-1024-<timestamp>/checkpoints/best-....ckpt'"
```

Sweep (3 architectures × 2 LR schedules = 6 runs):

```bash
uv run python scripts/train.py -m net=unet,unet_wide,unet_narrow lr=constant_1e4,cosine
```

Each run writes to `runs/<pipeline>-<net>-<dataset>-<timestamp>/`:

```
runs/<run>/
├── config.yaml         # resolved Hydra config (everything needed to rebuild the model)
├── checkpoints/        # best-<...>-val_dice=<...>.ckpt + last.ckpt
├── wandb/              # also streamed to wandb.ai/<entity>/patomorphology
├── .hydra/             # Hydra's bookkeeping
└── train.log
```

Both train loss (`train_loss`, DiceCELoss per tile) and full-image validation Dice (`val_dice`, sliding-window over `nmsc-2x`) stream to W&B. `val_dice` drives best-checkpoint selection and is the single segmentation-quality readout to compare across runs.

`notebooks/explore_run.ipynb` loads any run by path — rebuilds the model from `config.yaml`, reloads the best checkpoint, and visualizes predictions on the val set.

---

## Common commands

```bash
uv run pytest                            # run tests
uv run ruff check . && uv run ruff format .
uv run jupyter lab                       # notebooks/ uses the project venv
uv add <pkg>                             # add runtime dep
uv add --group dev <pkg>                 # add dev dep
WANDB_MODE=offline uv run python scripts/train.py   # record locally, don't sync
```

---

## Stack

- **uv** (package manager, pinned via `pyproject.toml` + `uv.lock`) · Python 3.12
- **PyTorch** + **Lightning** + **MONAI** (UNet, DiceCELoss, DiceMetric, sliding_window_inference)
- **Hydra** for config composition · **W&B** for experiment tracking
- **Pydantic v2** for every data class / DTO / config object (no `dataclasses`)
- **plotly** for visualization (not matplotlib)
