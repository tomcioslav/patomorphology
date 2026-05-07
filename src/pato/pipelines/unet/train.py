from datetime import datetime
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pato.pipelines.unet.config import UNetRunConfig
from pato.pipelines.unet.data import make_dataloaders
from pato.pipelines.unet.module import UNetLightning


def _default_run_name(config: UNetRunConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"unet-{config.source}-{config.source_root.name}-{timestamp}"


def train(
    config: UNetRunConfig,
    runs_dir: Path,
    run_name: str | None = None,
    fast_dev_run: bool = False,
) -> tuple[UNetLightning, L.Trainer, Path]:
    """Train a UNet according to `config`. Output goes to `runs_dir/<run_name>/`."""
    run_name = run_name or _default_run_name(config)
    run_path = runs_dir / run_name
    (run_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Persist the run definition next to the artefacts.
    (run_path / "config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))

    train_loader, val_loader = make_dataloaders(config)
    model = UNetLightning(num_classes=config.num_classes, learning_rate=config.learning_rate)

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
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, train_loader, val_loader)
    return model, trainer, run_path


if __name__ == "__main__":
    from config import paths

    cfg = UNetRunConfig(
        source="nmsc",
        source_root=paths.nmsc_5x,
    )
    train(cfg, runs_dir=paths.runs)
