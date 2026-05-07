import sys

import torch
import torch.utils.data

from pato.pipelines.unet.config import UNetRunConfig
from pato.schema import PatoImage
from pato.source import BaseImageMaskDataset, NMSCDataset, TiledDataset

# macOS uses `spawn` by default for DataLoader workers, which re-imports the
# whole module per worker and frequently dies on PyTorch + Pillow stacks.
# `fork` is faster and reliable here — Linux already defaults to it.
_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


def _build_source(config: UNetRunConfig) -> BaseImageMaskDataset:
    if config.source == "nmsc":
        return NMSCDataset(root=config.source_root)
    raise ValueError(f"Unknown source: {config.source!r}")


def make_dataloaders(
    config: UNetRunConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train/val DataLoaders with image-level split (no tile leakage)."""
    src = _build_source(config)

    g = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(len(src), generator=g).tolist()
    n_val = int(len(src) * config.val_ratio)
    val_src_idx = perm[:n_val]
    train_src_idx = perm[n_val:]

    train_ds = TiledDataset(
        source=src,
        target_size=config.target_size,
        overlap=config.overlap,
        source_indices=train_src_idx,
    )
    val_ds = TiledDataset(
        source=src,
        target_size=config.target_size,
        overlap=config.overlap,
        source_indices=val_src_idx,
    )

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=PatoImage.collate,
        persistent_workers=config.num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if config.num_workers > 0 else None,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
