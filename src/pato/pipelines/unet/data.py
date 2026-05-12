import sys

import torch
import torch.utils.data

from pato.dataset import DatasetViewer
from pato.pipelines.unet.config import UNetRunConfig
from pato.schema import PatoImage

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


def make_dataloaders(
    config: UNetRunConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train/val DataLoaders over a pre-tiled cache.

    `config.dataset_root` points at a UNet tile cache (built by
    `pato.pipelines.unet.preprocess.preprocess(...)`) — `metadata.json`
    has the splits, `samples/<id>.npz` each store one ready-to-train
    image+mask tile. Reading is a single small `.npz` per item.
    """
    train_src = DatasetViewer(root=config.dataset_root, split="train")
    val_src = DatasetViewer(root=config.dataset_root, split="val")

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=PatoImage.collate,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if config.num_workers > 0 else None,
    )
    train_loader = torch.utils.data.DataLoader(train_src, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_src, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
