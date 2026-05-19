"""Train dataloader for the UNet pipeline.

`DatasetViewer` is the inspection-time pydantic reader for any
`data/processed/<name>/`. We wrap it in `UNetDataset` so the torch-Dataset
interface (and its tensor-conversion contract) lives next to the pipeline
that consumes it — `DatasetViewer` itself stays inspection-only.

Validation is **not** a tile loader: see `pato.pipelines._val` for the
shared full-image val dataloader (sliding-window inference at val time).
"""

import sys
from pathlib import Path

import torch
import torch.utils.data

from pato.dataset import DatasetViewer

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


class UNetDataset(torch.utils.data.Dataset):
    """Tile cache → `(image, mask)` tensor pairs for UNet training.

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


def make_train_dataloader(
    dataset_root: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """Train DataLoader over a pre-tiled cache.

    `dataset_root` points at a tile cache (built by
    `pato.dataset.builders.TileBuilder(...).build(...)`) — `metadata.json`
    has the splits, `samples/<id>.npz` each store one ready-to-train
    image+mask tile. Bare cache names like `nmsc-2x-unet-512` are resolved
    to `data/processed/<name>` by DatasetViewer.
    """
    train_ds = UNetDataset(dataset_root, split="train")
    return torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if num_workers > 0 else None,
    )
