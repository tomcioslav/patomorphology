"""SAM-head pipeline.

`build(cfg, net)` is what the training script calls. The `sam_model`
name (which SAM encoder produced the cache) is read from the cache's
own `metadata.json`, so callers don't have to repeat themselves.
"""

from pathlib import Path
from typing import Any

import torch.nn as nn
from hydra.utils import instantiate

from pato.schema import DatasetMetadata


def build(cfg: Any, net: nn.Module):
    from pato.pipelines.sam_head.data import _resolve_cache_dir, make_dataloaders
    from pato.pipelines.sam_head.module import SAMHeadLightning

    cache_dir = _resolve_cache_dir(cfg.dataset.dataset_root)
    cache_meta = DatasetMetadata.model_validate_json(
        (cache_dir / "metadata.json").read_text()
    )
    sam_model = (cache_meta.config or {}).get("sam_model", "facebook/sam-vit-base")

    scheduler_partial = instantiate(cfg.lr.scheduler)
    lightning = SAMHeadLightning(
        model=net,
        sam_model=sam_model,
        learning_rate=cfg.lr.learning_rate,
        scheduler_partial=scheduler_partial,
    )
    train_loader, val_loader = make_dataloaders(
        dataset_root=cfg.dataset.dataset_root,
        batch_size=cfg.pipeline.batch_size,
        num_workers=cfg.pipeline.num_workers,
    )
    return lightning, train_loader, val_loader
