"""Notebook-facing helpers for working with training runs.

Each run is a folder under `paths.runs/` that contains a `config.yaml`
(the full resolved Hydra config), `checkpoints/`, and a local `wandb/`
cache mirroring what's synced to wandb.ai.

- `list_runs(runs_dir)` — names of all runs found
- `load_run(run_path)` — a `Run` object with the loaded config + paths
- `load_inference_model(run_path)` — reconstitute the trained model for
  inference (image→logits `nn.Module`). Net is rebuilt from `cfg.net` via
  Hydra `instantiate`; head weights come from the checkpoint.
- `load_predictor(run_path)` — wrap that model in a generic `Predictor`.
- `source_dataset_root(run)` — full-image source dataset path (from
  `cache.metadata.config["source_root"]`, fallback to `cfg.dataset.dataset_root`).
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
    if not runs_dir.exists():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if (p / "config.yaml").exists())


def list_runs_by_pipeline(runs_dir: Path, pipeline: str) -> list[str]:
    return [
        name
        for name in list_runs(runs_dir)
        if _pipeline_name(load_run(runs_dir / name)) == pipeline
    ]


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

    raise ValueError(f"Unknown pipeline: {pipeline!r}")


def load_predictor(run_path: Path):
    """Return a `pato.inference.Predictor` configured for this run.

    Pulls `target_size` / `overlap` from the run's cache metadata.
    """
    from pato.inference import Predictor

    run = load_run(run_path)
    model = load_inference_model(run_path)
    target_size, overlap = _resolve_tile_settings(run)
    return Predictor(model=model, target_size=target_size, overlap=overlap)


def source_dataset_root(run: Run) -> Path:
    """Return the upstream full-image dataset path for this run.

    Reads `source_root` from the cache's `metadata.config` and resolves
    it against the cache dir. Falls back to the run's `dataset_root` for
    runs that trained directly on a normalized dataset.
    """
    dataset_root = _resolved_dataset_root(run)
    cache_cfg = _cache_metadata_config(dataset_root)
    source_root = cache_cfg.get("source_root")
    if source_root is None:
        return dataset_root
    return (dataset_root / source_root).resolve()


def _pipeline_name(run: Run) -> str | None:
    pipeline = run.config.get("pipeline")
    if isinstance(pipeline, dict):
        return pipeline.get("name")
    return pipeline


def _resolved_dataset_root(run: Run) -> Path:
    """Mirror DatasetViewer's bare-name resolution so loaders see the
    same path no matter how `dataset_root` was written in the config.
    """
    raw = run.config["dataset"]["dataset_root"]
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


def _resolve_tile_settings(run: Run) -> tuple[int, int]:
    cache_cfg = _cache_metadata_config(_resolved_dataset_root(run))
    return int(cache_cfg["target_size"]), int(cache_cfg["overlap"])
