import sys

import torch
import torch.utils.data

from pato.dataset import DatasetViewer
from pato.pipelines.unet.config import UNetRunConfig
from pato.schema import PatoImage
from pato.utils.image_split import _tile_starts

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


class UNetDataset(torch.utils.data.Dataset):
    """UNet-specific: produce fixed-size tiles from a `DatasetViewer`.

    Tile coordinates are computed at init from `metadata.samples[id].size`
    (no pixel decompression). `dataset[i]` returns a `PatoImage` for the
    i-th tile. Tiling is owned by this pipeline; other pipelines (e.g.
    SAM-head) have different per-tile semantics and define their own
    Datasets in their own `data.py`.
    """

    def __init__(
        self,
        source: DatasetViewer,
        target_size: int,
        overlap: int,
    ):
        self.source = source
        self.target_size = target_size
        self.overlap = overlap

        index: list[tuple[int, int, int]] = []
        for src_idx, sid in enumerate(source.sample_ids):
            h, w = source.metadata.samples[sid].size
            for y in _tile_starts(h, target_size, overlap):
                for x in _tile_starts(w, target_size, overlap):
                    index.append((src_idx, y, x))
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> PatoImage:
        src_idx, y, x = self._index[idx]
        sample = self.source[src_idx]
        ts = self.target_size
        return PatoImage(
            image=sample.image[y : y + ts, x : x + ts],
            mask=sample.mask[y : y + ts, x : x + ts],
        )


def make_dataloaders(
    config: UNetRunConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train/val DataLoaders. Splits come from the dataset's metadata.json."""
    train_src = DatasetViewer(root=config.dataset_root, split="train")
    val_src = DatasetViewer(root=config.dataset_root, split="val")

    train_ds = UNetDataset(train_src, config.target_size, config.overlap)
    val_ds = UNetDataset(val_src, config.target_size, config.overlap)

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=PatoImage.collate,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if config.num_workers > 0 else None,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
