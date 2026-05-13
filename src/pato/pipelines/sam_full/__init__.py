"""SAM-full pipeline.

End-to-end fine-tuning of SAM encoder + decoder head. `build(cfg, net)`
takes a pre-instantiated `SAMSegmentation` and returns the LightningModule
plus train/val dataloaders.
"""

from typing import Any

import torch.nn as nn
from hydra.utils import instantiate


def build(cfg: Any, net: nn.Module):
    from pato.pipelines.sam_full.data import make_dataloaders
    from pato.pipelines.sam_full.module import SAMFullLightning

    scheduler_partial = instantiate(cfg.lr.scheduler)
    lightning = SAMFullLightning(
        model=net,
        learning_rate=cfg.lr.learning_rate,
        sam_learning_rate=cfg.pipeline.sam_learning_rate,
        scheduler_partial=scheduler_partial,
    )
    train_loader, val_loader = make_dataloaders(
        dataset_root=cfg.dataset.dataset_root,
        batch_size=cfg.pipeline.batch_size,
        num_workers=cfg.pipeline.num_workers,
    )
    return lightning, train_loader, val_loader
