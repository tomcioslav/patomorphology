"""UNet pipeline.

`build(cfg, net)` is what the training script calls. It takes the
resolved Hydra cfg and a pre-instantiated `nn.Module`, then returns
`(LightningModule, train_loader, val_loader)`.

Train uses the tile cache (`cfg.dataset.train`); val uses the normalized
full-image source named in `cfg.dataset.val`, with sliding-window
inference at the cache's `target_size`. See `pato.pipelines._val`.
"""

from typing import Any

import torch.nn as nn
from hydra.utils import instantiate

from pato.pipelines._val import (
    make_val_dataloader,
    read_cache_geometry,
)


def build(cfg: Any, net: nn.Module):
    from pato.pipelines.unet.data import make_train_dataloader
    from pato.pipelines.unet.module import UNetLightning

    target_size, overlap = read_cache_geometry(cfg.dataset.train)

    scheduler_partial = instantiate(cfg.lr.scheduler)
    lightning = UNetLightning(
        model=net,
        learning_rate=cfg.lr.learning_rate,
        scheduler_partial=scheduler_partial,
        val_target_size=target_size,
        val_overlap=overlap,
    )
    train_loader = make_train_dataloader(
        dataset_root=cfg.dataset.train,
        batch_size=cfg.pipeline.batch_size,
        num_workers=cfg.pipeline.num_workers,
    )
    val_loader = make_val_dataloader(
        val_source_root=cfg.dataset.val,
        num_workers=cfg.pipeline.num_workers,
    )
    return lightning, train_loader, val_loader
