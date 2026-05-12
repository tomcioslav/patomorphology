from typing import Literal

from pato.pipelines.base import BaseRunConfig


class SAMFullRunConfig(BaseRunConfig):
    """Everything that fully describes a `sam_full` training run.

    Unlike `sam_head`, this pipeline trains the SAM encoder **and** the
    decoder head end-to-end on raw image tiles. No feature cache —
    `dataset_root` points at a normalized dataset (e.g. `nmsc-2x`) and
    tiling happens on the fly. Two optimizer groups: SAM at a small
    learning rate, head at the larger one.
    """

    pipeline: Literal["sam_full"] = "sam_full"
    max_epochs: int = 10

    sam_model: str = "facebook/sam-vit-base"
    target_size: int = 1024
    overlap: int = 64
    mask_pad_class: int = 8

    batch_size: int = 2
    num_workers: int = 4

    learning_rate: float = 3e-4
    sam_learning_rate: float = 1e-6

    num_classes: int = 12
    feature_channels: int = 256
    head_widths: tuple[int, ...] = (128, 64, 32, 16)
