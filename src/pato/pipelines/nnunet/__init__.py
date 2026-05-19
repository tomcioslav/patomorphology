"""nnU-Net-style pipeline.

Same shape as the baseline UNet pipeline (tile cache for train, normalized
source for sliding-window val), but with three recipe changes:

- `DynUNet` architecture with deep supervision (see `conf/net/nnunet.yaml`).
- Heavy on-the-fly augmentation (see `pato.pipelines.nnunet.data`).
- SGD + Nesterov(momentum=0.99) + (typically) PolyLR — pin the LR group
  with `lr=poly` to get nnU-Net's full schedule.

Pair with `dataset=nmsc-2x-unet-512` for a clean A/B against
`pipeline=unet` net=unet`. Same tiles, same val source — only the recipe
differs.
"""

from typing import Any

import torch.nn as nn
from hydra.utils import instantiate

from pato.pipelines._val import (
    make_val_dataloader,
    read_cache_geometry,
)


def build(cfg: Any, net: nn.Module):
    from pato.pipelines.nnunet.data import make_train_dataloader
    from pato.pipelines.nnunet.module import NNUNetLightning

    target_size, overlap = read_cache_geometry(cfg.dataset.train)

    scheduler_partial = instantiate(cfg.lr.scheduler)
    lightning = NNUNetLightning(
        model=net,
        learning_rate=cfg.lr.learning_rate,
        scheduler_partial=scheduler_partial,
        val_target_size=target_size,
        val_overlap=overlap,
        momentum=cfg.pipeline.momentum,
        weight_decay=cfg.pipeline.weight_decay,
    )
    train_loader = make_train_dataloader(
        dataset_root=cfg.dataset.train,
        batch_size=cfg.pipeline.batch_size,
        num_workers=cfg.pipeline.num_workers,
        augment=cfg.pipeline.augment,
    )
    val_loader = make_val_dataloader(
        val_source_root=cfg.dataset.val,
        num_workers=cfg.pipeline.num_workers,
    )
    return lightning, train_loader, val_loader
