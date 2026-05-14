"""Unified SAM segmentation pipeline (frozen head-only + end-to-end).

`build(cfg, net)` reads `cfg.pipeline.sam_frozen` to pick the training
regime and the matching dataloader factory. `net` is always a
`SAMSegmentation` (encoder + head) — see `conf/net/sam*.yaml`.
"""

from typing import Any

import torch.nn as nn
from hydra.utils import instantiate


def build(cfg: Any, net: nn.Module):
    from pato.pipelines.sam import data
    from pato.pipelines.sam.module import SAMLightning

    sam_frozen = bool(cfg.pipeline.sam_frozen)
    scheduler_partial = instantiate(cfg.lr.scheduler)
    lightning = SAMLightning(
        model=net,
        sam_frozen=sam_frozen,
        learning_rate=cfg.lr.learning_rate,
        sam_learning_rate=cfg.pipeline.sam_learning_rate,
        scheduler_partial=scheduler_partial,
    )

    make = data.make_feature_dataloaders if sam_frozen else data.make_tile_dataloaders
    train_loader, val_loader = make(
        cfg.dataset.dataset_root,
        batch_size=cfg.pipeline.batch_size,
        num_workers=cfg.pipeline.num_workers,
    )
    return lightning, train_loader, val_loader
