"""Unified SAM segmentation pipeline (frozen head-only + end-to-end).

`build(cfg, net)` reads `cfg.pipeline.sam_frozen` to pick the **training**
regime and the matching dataloader factory. `net` is always a
`SAMSegmentation` (encoder + head) — see `conf/net/sam*.yaml`.

Validation is identical across both regimes: full-image sliding-window
inference over the normalized source dataset named in `cfg.dataset.val`.
The frozen/unfrozen split is a train-time-only optimization; at val time
we always run the full encoder+head — see `pato.pipelines._val`.
"""

from typing import Any

import torch.nn as nn
from hydra.utils import instantiate

from pato.pipelines._val import (
    make_val_dataloader,
    read_cache_geometry,
)


def build(cfg: Any, net: nn.Module):
    from pato.pipelines.sam import data
    from pato.pipelines.sam.module import SAMLightning

    sam_frozen = bool(cfg.pipeline.sam_frozen)
    target_size, overlap = read_cache_geometry(cfg.dataset.train)

    scheduler_partial = instantiate(cfg.lr.scheduler)
    lightning = SAMLightning(
        model=net,
        sam_frozen=sam_frozen,
        learning_rate=cfg.lr.learning_rate,
        sam_learning_rate=cfg.pipeline.sam_learning_rate,
        gradient_checkpointing=bool(cfg.pipeline.get("gradient_checkpointing", False)),
        scheduler_partial=scheduler_partial,
        val_target_size=target_size,
        val_overlap=overlap,
    )

    make_train = (
        data.make_feature_train_dataloader
        if sam_frozen
        else data.make_tile_train_dataloader
    )
    train_loader = make_train(
        cfg.dataset.train,
        batch_size=cfg.pipeline.batch_size,
        num_workers=cfg.pipeline.num_workers,
    )
    val_loader = make_val_dataloader(
        val_source_root=cfg.dataset.val,
        num_workers=cfg.pipeline.num_workers,
    )
    return lightning, train_loader, val_loader
