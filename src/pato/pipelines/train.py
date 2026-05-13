"""Pipeline-agnostic training entry point.

`train(cfg, runs_dir)`:
  1. Instantiates the net via Hydra (`cfg.net._target_`).
  2. Imports `pato.pipelines.<cfg.pipeline.name>` and calls its
     `build(cfg, net)` to get `(LightningModule, train_loader, val_loader)`.
  3. Sets up Trainer (checkpointing, W&B logging, run-config save) — the
     same setup for every pipeline.

Adding a new pipeline = drop a package under `pato/pipelines/<name>/`
with a `build(cfg, net)` function. Nothing in this file changes.
"""

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

# Tensor-core friendly matmul on Ampere+ NVIDIA GPUs.
torch.set_float32_matmul_precision("high")


def _default_run_name(cfg) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{cfg.pipeline.name}-{Path(str(cfg.dataset.dataset_root)).name}-{timestamp}"


def warm_start_model(lightning_module: L.LightningModule, ckpt_path: Path) -> str:
    """Initialize `lightning_module.model` from a previous run's checkpoint.

    Warm-start = load model weights only, no optimizer/scheduler state.
    Two cases handled:

    - **Same architecture** — direct `model.load_state_dict(..., strict=True)`.
      Works for UNet→UNet, sam_head→sam_head, sam_full→sam_full.
    - **sam_head → sam_full** — the source ckpt holds a `SAMSegHead`; the
      target's `.model` is a `SAMSegmentation` with a `.head` sub-module of
      the same type. We load the source state into `.model.head`, leaving
      the encoder at its HF-pretrained init.

    Returns a short tag ("direct" / "head") describing which path was used.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_state = ckpt["state_dict"]
    # Strip the `model.` prefix Lightning adds when self.model is registered.
    model_state = {
        k[len("model."):]: v for k, v in full_state.items() if k.startswith("model.")
    }
    if not model_state:
        raise RuntimeError(
            f"No `model.*` keys in checkpoint {ckpt_path}; nothing to warm-start."
        )

    try:
        lightning_module.model.load_state_dict(model_state, strict=True)
        return "direct"
    except RuntimeError:
        pass

    # Cross-pipeline: source is a SAMSegHead, target wraps a SAMSegHead at .head
    if hasattr(lightning_module.model, "head") and hasattr(
        lightning_module.model, "encoder"
    ):
        try:
            lightning_module.model.head.load_state_dict(model_state, strict=True)
            return "head"
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not warm-start {type(lightning_module.model).__name__} "
                f"from {ckpt_path}. Tried direct load and head-only routing. "
                f"Final error: {e}"
            ) from e

    raise RuntimeError(
        f"Could not warm-start {type(lightning_module.model).__name__} "
        f"from {ckpt_path}. Source/target architectures don't match and no "
        f"cross-pipeline routing applies."
    )


def train(
    cfg: Any,
    runs_dir: Path,
    run_name: str | None = None,
) -> tuple[L.LightningModule, L.Trainer, Path]:
    """Train any pipeline. Output → `runs_dir/<run_name>/`."""
    run_name = run_name or _default_run_name(cfg)
    run_path = runs_dir / run_name
    (run_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=run_path / "config.yaml", resolve=True)

    net = instantiate(cfg.net)
    pipeline_pkg = importlib.import_module(f"pato.pipelines.{cfg.pipeline.name}")
    lightning, train_loader, val_loader = pipeline_pkg.build(cfg, net=net)

    init_ckpt = cfg.get("init_from_checkpoint")
    if init_ckpt:
        tag = warm_start_model(lightning, Path(init_ckpt))
        print(f"Warm-started {type(lightning.model).__name__} ({tag}) from {init_ckpt}")

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
    logger = WandbLogger(
        project="patomorphology",
        name=run_name,
        save_dir=str(run_path),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        precision=cfg.pipeline.precision,
        benchmark=True,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=cfg.fast_dev_run,
    )
    trainer.fit(lightning, train_loader, val_loader)
    return lightning, trainer, run_path
