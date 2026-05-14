"""Dataloaders for the unified SAM pipeline.

Two factories — `build()` picks one based on `sam_frozen`:

- `make_feature_dataloaders` — frozen mode. Reads a SAM-feature cache
  (`SAMFeatureBuilder` output); yields `(features, mask)`.
- `make_tile_dataloaders` — end-to-end mode. Reads a raw 1024-tile cache
  (`TileBuilder(target_size=1024)` output); yields `(image, mask)`.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from pato.dataset import DatasetViewer
from pato.schema import DatasetMetadata, PatoImage

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None
_DEFAULT_DATA_PROCESSED = Path("data/processed")


def _resolve_cache_dir(v: str | Path) -> Path:
    p = Path(v)
    if not p.is_absolute() and len(p.parts) == 1:
        return _DEFAULT_DATA_PROCESSED / p
    return p


def _loader_kwargs(batch_size: int, num_workers: int, collate_fn=None) -> dict:
    return dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if num_workers > 0 else None,
    )


class SAMFeatureDataset(torch.utils.data.Dataset):
    """Reads a SAM-feature cache. `dataset[i]` → `(features, mask)`:
    features `(256, 64, 64) float32`, mask `(H, W) int64`.
    """

    def __init__(self, cache_dir: str | Path, split: str | None = None):
        self.cache_dir = _resolve_cache_dir(cache_dir)
        meta_path = self.cache_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata.json at {self.cache_dir}. Build the cache first."
            )
        self.metadata = DatasetMetadata.model_validate_json(meta_path.read_text())

        if split is None:
            self._tile_ids = sorted(self.metadata.samples.keys())
        else:
            if split not in self.metadata.splits:
                raise ValueError(
                    f"split {split!r} not in metadata splits "
                    f"({sorted(self.metadata.splits.keys())})"
                )
            self._tile_ids = list(self.metadata.splits[split])

    def __len__(self) -> int:
        return len(self._tile_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_id = self._tile_ids[idx]
        sample_meta = self.metadata.samples[tile_id]
        with np.load(self.cache_dir / sample_meta.path) as data:
            features = torch.from_numpy(data["image"])              # (256, 64, 64)
            mask = torch.from_numpy(data["mask"].astype(np.int64))  # (H, W)
        return features, mask


def make_feature_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Frozen-SAM training: train/val loaders over a SAM-feature cache."""
    train_ds = SAMFeatureDataset(dataset_root, split="train")
    val_ds = SAMFeatureDataset(dataset_root, split="val")
    kw = _loader_kwargs(batch_size, num_workers)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **kw)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **kw)
    return train_loader, val_loader


def make_tile_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 4,
    num_workers: int = 8,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """End-to-end training: train/val loaders over a raw 1024-tile cache.
    Yields `(image, mask)` via `PatoImage.collate`.
    """
    train_src = DatasetViewer(root=dataset_root, split="train")
    val_src = DatasetViewer(root=dataset_root, split="val")
    kw = _loader_kwargs(batch_size, num_workers, collate_fn=PatoImage.collate)
    train_loader = torch.utils.data.DataLoader(train_src, shuffle=True, **kw)
    val_loader = torch.utils.data.DataLoader(val_src, shuffle=False, **kw)
    return train_loader, val_loader
