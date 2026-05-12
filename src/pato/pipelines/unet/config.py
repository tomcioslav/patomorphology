from typing import Literal

from pato.pipelines.base import BaseRunConfig


class UNetRunConfig(BaseRunConfig):
    """Everything that fully describes a UNet training run.

    Saved as `config.yaml` next to checkpoints. `pato.experiments.load_predictor`
    reads `pipeline` to dispatch to `UNetPredictor`; the predictor reads the
    rest to reconstitute training-time tile size / overlap / etc. exactly.
    """

    pipeline: Literal["unet"] = "unet"

    target_size: int = 512
    overlap: int = 64

    batch_size: int = 4
    num_workers: int = 4

    learning_rate: float = 1e-4

    num_classes: int = 2
    channels: tuple[int, ...] = (32, 64, 128, 256, 512)
    num_res_units: int = 2
