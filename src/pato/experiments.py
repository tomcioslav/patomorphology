"""Notebook-facing helpers for working with training runs.

Each run is a folder under `paths.runs/` that contains a `config.yaml`
(the full resolved Hydra config), `checkpoints/`, and a local `wandb/`
cache mirroring what's synced to wandb.ai.

- `list_runs(runs_dir)` — names of all runs found
- `list_datasets(data_processed_dir)` — names of normalized full-image
  datasets (excludes pipeline tile / feature caches).
- `load_run(run_path)` — a `Run` object with the loaded config + paths
- `load_inference_model(run_path)` — reconstitute the trained model for
  inference (image→logits `nn.Module`). Net is rebuilt from `cfg.net` via
  Hydra `instantiate`; head weights come from the checkpoint.
- `load_predictor(run_path)` — wrap that model in a generic `Predictor`.
  Tile size / overlap come from the training cache when it's still on
  disk, else from the net's intrinsic input size — so a predictor builds
  even when the cache lives on another machine.
"""

from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml
from hydra.utils import instantiate
from pydantic import BaseModel

from pato.schema import DatasetMetadata

_DEFAULT_DATA_PROCESSED = Path("data/processed")


class Run(BaseModel):
    """A single training run on disk."""

    name: str
    path: Path
    config: dict[str, Any]

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    def best_checkpoint(self) -> Path | None:
        candidates = sorted(
            p for p in self.checkpoints_dir.glob("*.ckpt") if p.name != "last.ckpt"
        )
        return candidates[0] if candidates else None

    def last_checkpoint(self) -> Path:
        return self.checkpoints_dir / "last.ckpt"


def list_runs(runs_dir: Path) -> list[str]:
    """Run names under `runs_dir`, **oldest first / newest last**.

    Sorted by directory mtime, not name: the run-name format
    (`<pipeline>-<net>-<dataset>-<timestamp>`) doesn't lead with the
    timestamp, so a lexical sort groups by architecture instead of time.
    `list_runs(...)[-1]` is therefore the most recent run.
    """
    if not runs_dir.exists():
        return []
    dirs = [p for p in runs_dir.iterdir() if (p / "config.yaml").exists()]
    dirs.sort(key=lambda p: p.stat().st_mtime)
    return [p.name for p in dirs]


def list_runs_by_pipeline(runs_dir: Path, pipeline: str) -> list[str]:
    return [
        name
        for name in list_runs(runs_dir)
        if _pipeline_name(load_run(runs_dir / name)) == pipeline
    ]


def list_datasets(data_processed_dir: Path) -> list[str]:
    """Names of **normalized full-image** datasets under `data_processed_dir`.

    These hold full images + masks (no tiling) and are what the run-analysis
    notebook predicts against — chosen independently of which cache a run
    happened to train on.

    Pipeline caches (tile / SAM-feature dirs) are filtered out: being
    tiled, they record a `target_size` in `metadata.config`; full-image
    datasets never do.
    """
    if not data_processed_dir.exists():
        return []
    names = []
    for p in sorted(data_processed_dir.iterdir()):
        if not (p / "metadata.json").exists():
            continue
        if "target_size" not in _cache_metadata_config(p):
            names.append(p.name)
    return names


def load_run(run_path: Path) -> Run:
    cfg_path = run_path / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml at {run_path}")
    return Run(
        name=run_path.name,
        path=run_path,
        config=yaml.safe_load(cfg_path.read_text()),
    )


def load_inference_model(run_path: Path) -> nn.Module:
    """Reconstitute the trained model as an image→logits `nn.Module`.

    Instantiates the net from `cfg.net` (same Hydra config used at train
    time), loads the checkpoint into the right Lightning module with
    `model=net`, then calls its `to_inference_model()` to handle the
    train→infer asymmetry (e.g. sam_head reattaching a frozen encoder).
    """
    run = load_run(run_path)
    pipeline = _pipeline_name(run)
    ckpt = run.best_checkpoint() or run.last_checkpoint()
    net = instantiate(run.config["net"])

    if pipeline == "unet":
        from pato.pipelines.unet.module import UNetLightning

        return UNetLightning.load_from_checkpoint(ckpt, model=net).to_inference_model()

    if pipeline == "sam":
        from pato.pipelines.sam.module import SAMLightning

        return SAMLightning.load_from_checkpoint(ckpt, model=net).to_inference_model()

    if pipeline == "nnunet":
        from pato.pipelines.nnunet.module import NNUNetLightning

        return NNUNetLightning.load_from_checkpoint(
            ckpt, model=net
        ).to_inference_model()

    raise ValueError(f"Unknown pipeline: {pipeline!r}")


def load_predictor(
    run_path: Path,
    *,
    target_size: int | None = None,
    overlap: int | None = None,
):
    """Return a `pato.inference.Predictor` configured for this run.

    `target_size` / `overlap` default to the run's cache metadata when the
    cache is still on disk, else to the net's intrinsic input size (see
    `_resolve_tile_settings`). Pass either explicitly to override.
    """
    from pato.inference import Predictor

    run = load_run(run_path)
    model = load_inference_model(run_path)
    resolved_size, resolved_overlap = _resolve_tile_settings(run)
    return Predictor(
        model=model,
        target_size=resolved_size if target_size is None else target_size,
        overlap=resolved_overlap if overlap is None else overlap,
    )


def _pipeline_name(run: Run) -> str | None:
    pipeline = run.config.get("pipeline")
    if isinstance(pipeline, dict):
        return pipeline.get("name")
    return pipeline


def _resolved_dataset_root(run: Run) -> Path:
    """Path to the run's **train** cache, resolved like `DatasetViewer`
    does (bare names like `nmsc-2x-unet-512` → `data/processed/<name>`).

    Supports both the current `dataset.train` field and the legacy
    `dataset.dataset_root` field that pre-train/val-split runs wrote.
    """
    ds = run.config["dataset"]
    raw = ds.get("train", ds.get("dataset_root"))
    if raw is None:
        raise KeyError(
            f"Run {run.path.name} config has no `dataset.train` "
            f"(or legacy `dataset.dataset_root`)."
        )
    p = Path(str(raw))
    if not p.is_absolute() and len(p.parts) == 1:
        return _DEFAULT_DATA_PROCESSED / p
    return p


def _cache_metadata_config(dataset_root: Path) -> dict[str, Any]:
    meta_path = dataset_root / "metadata.json"
    if not meta_path.exists():
        return {}
    meta = DatasetMetadata.model_validate_json(meta_path.read_text())
    return meta.config or {}


_DEFAULT_OVERLAP = 64


def _net_target_sizes() -> dict[str, int]:
    """Intrinsic sliding-window ROI size per net, keyed by a substring of
    the net `_target_`. Used when the training cache isn't on disk to read
    it from. SAM's encoder is hard-fixed at 1024; UNet is fully
    convolutional, so 512 is just the cache convention, not a constraint.

    Imported lazily — `SAMSegmentation` pulls in `transformers`, and the
    rest of this module stays import-light on purpose.
    """
    from pato.components.models.sam_segmentation import SAMSegmentation

    return {
        "SAMSegmentation": SAMSegmentation.IMAGE_SIZE,
        "UNet": 512,
        "DynUNet": 512,
    }


def _resolve_tile_settings(run: Run) -> tuple[int, int]:
    """Resolve sliding-window `(target_size, overlap)` for the run's model.

    Prefers the training cache's `metadata.config` when that cache is still
    on disk; otherwise falls back to the net's intrinsic input size, so a
    predictor builds even when the cache lives on another machine.
    """
    cache_cfg = _cache_metadata_config(_resolved_dataset_root(run))
    overlap = int(cache_cfg.get("overlap", _DEFAULT_OVERLAP))
    if "target_size" in cache_cfg:
        return int(cache_cfg["target_size"]), overlap

    net_target = run.config["net"]["_target_"]
    for key, size in _net_target_sizes().items():
        if key in net_target:
            return size, overlap
    raise ValueError(
        f"Cannot resolve target_size for net {net_target!r}: the training "
        "cache is not on disk and the net has no known default. Pass "
        "target_size= to load_predictor()."
    )
