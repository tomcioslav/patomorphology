from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class UNetRunConfig(BaseModel):
    """Everything that fully describes a UNet training run.

    Saved as `config.yaml` next to checkpoints. `pato.experiments.load_predictor`
    reads `pipeline` to dispatch to `UNetPredictor`; the predictor reads the
    rest to reconstitute training-time tile size / overlap / etc. exactly.
    """

    pipeline: Literal["unet"] = "unet"

    # Source dataset
    source: Literal["nmsc"] = "nmsc"  # discriminator for `_build_source`
    source_root: Path                 # absolute path to the chosen source root

    # Tiling
    target_size: int = 512
    overlap: int = 64

    # Data split / loader
    val_ratio: float = 0.2
    batch_size: int = 4
    num_workers: int = 4
    seed: int = 42

    # Optimization
    learning_rate: float = 1e-4
    max_epochs: int = 5

    # Model
    num_classes: int = 12
