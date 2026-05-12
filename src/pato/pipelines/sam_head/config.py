from typing import Literal

from pato.pipelines.base import BaseRunConfig


class SAMHeadRunConfig(BaseRunConfig):
    """Everything that fully describes a SAM-head training run.

    `dataset_root` points at a SAM-feature cache (a directory under
    `data/processed/<name>-sam-...-<tile>/`). The cache itself describes how
    it was produced via `metadata.config` (sam_model, target_size, overlap,
    mask_pad_class) — those are dataset properties, not run-config knobs.
    """

    pipeline: Literal["sam_head"] = "sam_head"
    max_epochs: int = 20

    batch_size: int = 8
    num_workers: int = 4

    learning_rate: float = 3e-4

    num_classes: int = 12
    feature_channels: int = 256
    head_widths: tuple[int, ...] = (128, 64, 32, 16)
