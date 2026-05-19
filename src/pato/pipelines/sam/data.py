"""Train dataloaders for the unified SAM pipeline.

Two factories — `build()` picks one based on `sam_frozen`:

- `make_feature_train_dataloader` — frozen mode. Reads a SAM-feature cache
  (`SAMFeatureBuilder` output); yields `(features, mask)`.
- `make_tile_train_dataloader` — end-to-end mode. Reads a raw 1024-tile
  cache (`TileBuilder(target_size=1024)` output); yields `(image, mask)`.

Validation is the same for both regimes (full-image sliding window) and
lives in `pato.pipelines._val` — see `SAMLightning.validation_step`.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from pato.dataset import DatasetViewer
from pato.schema import DatasetMetadata

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None
_DEFAULT_DATA_PROCESSED = Path("data/processed")


def _resolve_cache_dir(v: str | Path) -> Path:
    p = Path(v)
    if not p.is_absolute() and len(p.parts) == 1:
        return _DEFAULT_DATA_PROCESSED / p
    return p


def _loader_kwargs(batch_size: int, num_workers: int) -> dict:
    return dict(
        batch_size=batch_size,
        num_workers=num_workers,
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


class SAMTileDataset(torch.utils.data.Dataset):
    """Raw 1024-tile cache → `(image, mask)` tensor pairs for end-to-end SAM.

    Thin torch wrapper around `DatasetViewer`. Each item is the result
    of `PatoImage.to_torch()`:
        image: (3, H, W) float32 in [0, 1]
        mask:  (H, W)    int64 class indices
    """

    def __init__(self, dataset_root: str | Path, split: str | None = None):
        self._viewer = DatasetViewer(root=dataset_root, split=split)

    def __len__(self) -> int:
        return len(self._viewer)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._viewer[idx].to_torch()


def make_feature_train_dataloader(
    dataset_root: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """Frozen-SAM training: train loader over a SAM-feature cache."""
    train_ds = SAMFeatureDataset(dataset_root, split="train")
    return torch.utils.data.DataLoader(
        train_ds, shuffle=True, **_loader_kwargs(batch_size, num_workers)
    )


def make_tile_train_dataloader(
    dataset_root: str | Path,
    batch_size: int = 4,
    num_workers: int = 8,
) -> torch.utils.data.DataLoader:
    """End-to-end training: train loader over a raw 1024-tile cache."""
    train_ds = SAMTileDataset(dataset_root, split="train")
    return torch.utils.data.DataLoader(
        train_ds, shuffle=True, **_loader_kwargs(batch_size, num_workers)
    )
