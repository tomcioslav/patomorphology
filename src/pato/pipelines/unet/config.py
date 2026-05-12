from typing import Literal

from pato.pipelines.base import BaseRunConfig


class UNetRunConfig(BaseRunConfig):
    """Everything that fully describes a UNet training run.

    `dataset_root` points at a UNet tile cache (built by
    `pato.pipelines.unet.preprocess.preprocess`). The cache itself
    describes how it was built in `metadata.config` (target_size,
    overlap, mask_pad_class) — those are properties of the cache, not
    run-config knobs.
    """

    pipeline: Literal["unet"] = "unet"

    batch_size: int = 4
    num_workers: int = 4

    learning_rate: float = 1e-4

    num_classes: int = 2
    channels: tuple[int, ...] = (32, 64, 128, 256, 512)
    num_res_units: int = 2
