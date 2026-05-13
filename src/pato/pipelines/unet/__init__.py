"""UNet pipeline.

`build(cfg, net)` is what the training script calls. It takes the
resolved Hydra cfg and a pre-instantiated `nn.Module`, then returns
`(LightningModule, train_loader, val_loader)`.
"""

from typing import Any

import torch.nn as nn
from hydra.utils import instantiate


def build(cfg: Any, net: nn.Module):
    from pato.pipelines.unet.data import make_dataloaders
    from pato.pipelines.unet.module import UNetLightning

    scheduler_partial = instantiate(cfg.lr.scheduler)
    lightning = UNetLightning(
        model=net,
        learning_rate=cfg.lr.learning_rate,
        scheduler_partial=scheduler_partial,
    )
    train_loader, val_loader = make_dataloaders(
        dataset_root=cfg.dataset.dataset_root,
        batch_size=cfg.pipeline.batch_size,
        num_workers=cfg.pipeline.num_workers,
    )
    return lightning, train_loader, val_loader
