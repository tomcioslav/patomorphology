import sys
from pathlib import Path

import torch
import torch.utils.data

from pato.dataset import DatasetViewer
from pato.schema import PatoImage

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


def make_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train/val DataLoaders over a pre-tiled cache.

    `dataset_root` points at a tile cache (built by
    `pato.dataset.builders.TileBuilder(...).build(...)`) — `metadata.json`
    has the splits, `samples/<id>.npz` each store one ready-to-train
    image+mask tile. Bare cache names like `nmsc-2x-unet-512` are resolved
    to `data/processed/<name>` by DatasetViewer.
    """
    train_src = DatasetViewer(root=dataset_root, split="train")
    val_src = DatasetViewer(root=dataset_root, split="val")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=PatoImage.collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if num_workers > 0 else None,
    )
    train_loader = torch.utils.data.DataLoader(train_src, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_src, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
