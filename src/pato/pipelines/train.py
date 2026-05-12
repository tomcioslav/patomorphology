"""Pipeline-agnostic training entry point.

`train(config, runs_dir)` reads `config.pipeline`, dynamically imports
`pato.pipelines.<config.pipeline>`, and asks that package's `build(config)`
factory for the LightningModule + dataloaders. The Trainer setup
(checkpointing, TensorBoard logging, config.yaml persistence) is identical
for every pipeline and lives here.

Adding a new pipeline never requires editing this file: drop a new
package under `pato/pipelines/<name>/` with:
  - a `RunConfig` subclass of `BaseRunConfig` whose `pipeline` field
    matches the package name,
  - a `build(config)` function in `__init__.py`.

Hydra-friendly: when you switch to Hydra you can replace the importlib
hop with `hydra.utils.instantiate` on a `_target_` field — the rest of
this file stays the same.
"""

import importlib
from datetime import datetime
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pato.pipelines.base import BaseRunConfig


def _default_run_name(config: BaseRunConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{config.pipeline}-{config.dataset_root.name}-{timestamp}"


def _build_pipeline(config: BaseRunConfig):
    """Resolve the pipeline by name and call its `build(config)` factory."""
    pkg = importlib.import_module(f"pato.pipelines.{config.pipeline}")
    if not hasattr(pkg, "build"):
        raise AttributeError(
            f"pato.pipelines.{config.pipeline} must export a `build(config)` "
            "function in its __init__.py"
        )
    return pkg.build(config)


def train(
    config: BaseRunConfig,
    runs_dir: Path,
    run_name: str | None = None,
    fast_dev_run: bool = False,
) -> tuple[L.LightningModule, L.Trainer, Path]:
    """Train any pipeline. Output → `runs_dir/<run_name>/`."""
    run_name = run_name or _default_run_name(config)
    run_path = runs_dir / run_name
    (run_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_path / "config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))

    model, train_loader, val_loader = _build_pipeline(config)

    callbacks = [
        ModelCheckpoint(
            dirpath=run_path / "checkpoints",
            filename="best-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    ]
    logger = TensorBoardLogger(save_dir=str(run_path), name="tensorboard", version="")

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        precision=config.precision,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, train_loader, val_loader)
    return model, trainer, run_path
