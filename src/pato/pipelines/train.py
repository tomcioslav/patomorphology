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
from pathlib import Path
from typing import Any

import lightning as L
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("high")


def warm_start_model(lightning_module: L.LightningModule, ckpt_path: Path) -> None:
    """Initialize `lightning_module.model` from a previous run's checkpoint.

    Warm-start = load model weights only; no optimizer/scheduler state.
    A plain `strict=True` load of the `model.*` keys — source and target
    must share architecture (the same `conf/net/*` config).

    For the `sam` pipeline this works whether the source run was frozen
    (`sam_frozen=True`) or end-to-end (`sam_frozen=False`): `self.model` is
    always a full `SAMSegmentation`, so every checkpoint carries the same
    `model.encoder.* + model.head.*` keys. Warm-starting an unfrozen run
    from a frozen run's head-only checkpoint is therefore a direct load —
    the encoder weights come along as the (unchanged) HF-pretrained values.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    if not model_state:
        raise RuntimeError(
            f"No `model.*` keys in checkpoint {ckpt_path}; nothing to warm-start."
        )
    try:
        lightning_module.model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not warm-start {type(lightning_module.model).__name__} from "
            f"{ckpt_path}. Source and target net architectures must match "
            f"(same conf/net/* config). Underlying error: {e}"
        ) from e


def train(
    cfg: Any,
    run_dir: Path,
) -> tuple[L.LightningModule, L.Trainer, Path]:
    """Train any pipeline. All artifacts go into `run_dir` (the directory
    Hydra created for this job — see the `hydra.run.dir` / `hydra.sweep`
    block in `conf/config.yaml`). `train()` writes `config.yaml`,
    `checkpoints/`, and the local `wandb/` cache alongside Hydra's own
    `.hydra/` + `train.log`.
    """
    run_path = Path(run_dir)
    (run_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=run_path / "config.yaml", resolve=True)

    net = instantiate(cfg.net)
    pipeline_pkg = importlib.import_module(f"pato.pipelines.{cfg.pipeline.name}")
    lightning, train_loader, val_loader = pipeline_pkg.build(cfg, net=net)

    init_ckpt = cfg.get("init_from_checkpoint")
    if init_ckpt:
        warm_start_model(lightning, Path(init_ckpt))
        print(f"Warm-started {type(lightning.model).__name__} from {init_ckpt}")

    callbacks = [
        ModelCheckpoint(
            dirpath=run_path / "checkpoints",
            filename="best-{epoch:02d}-{val_dice:.3f}",
            monitor="val_dice",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
    ]
    logger = WandbLogger(
        project="patomorphology",
        name=run_path.name,
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
    try:
        trainer.fit(lightning, train_loader, val_loader)
    finally:
        if wandb.run is not None:
            wandb.finish()
    return lightning, trainer, run_path
