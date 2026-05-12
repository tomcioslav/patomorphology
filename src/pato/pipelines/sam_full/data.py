import sys

import numpy as np
import torch
import torch.utils.data

from pato.dataset import DatasetViewer
from pato.pipelines.sam_full.config import SAMFullRunConfig
from pato.schema import PatoImage
from pato.utils.image_split import _tile_starts

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


def _pad_to_min(image: np.ndarray, mask: np.ndarray, min_size: int, mask_pad_class: int):
    h, w = image.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=mask_pad_class)
    return image, mask


class SAMFullDataset(torch.utils.data.Dataset):
    """On-the-fly 1024×1024 tile producer for end-to-end SAM training.

    Wraps a `DatasetViewer` (a normalized dataset). Tile coordinates are
    computed at init from `metadata.samples[id].size` (cheap — no pixel
    decompression). `dataset[i]` decompresses the parent image, pads it to
    at least `target_size` if needed, slices, and returns a `PatoImage`.
    Use `PatoImage.collate` as the DataLoader `collate_fn`.
    """

    def __init__(
        self,
        source: DatasetViewer,
        target_size: int,
        overlap: int,
        mask_pad_class: int,
    ):
        self.source = source
        self.target_size = target_size
        self.overlap = overlap
        self.mask_pad_class = mask_pad_class

        index: list[tuple[int, int, int]] = []
        for src_idx, sid in enumerate(source.sample_ids):
            h, w = source.metadata.samples[sid].size
            h_padded = max(h, target_size)
            w_padded = max(w, target_size)
            for y in _tile_starts(h_padded, target_size, overlap):
                for x in _tile_starts(w_padded, target_size, overlap):
                    index.append((src_idx, y, x))
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> PatoImage:
        src_idx, y, x = self._index[idx]
        sample = self.source[src_idx]
        image, mask = _pad_to_min(sample.image, sample.mask, self.target_size, self.mask_pad_class)
        ts = self.target_size
        return PatoImage(
            image=image[y : y + ts, x : x + ts],
            mask=mask[y : y + ts, x : x + ts],
        )


def make_dataloaders(
    config: SAMFullRunConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train/val DataLoaders. Splits come from the dataset's metadata.json."""
    train_src = DatasetViewer(root=config.dataset_root, split="train")
    val_src = DatasetViewer(root=config.dataset_root, split="val")

    train_ds = SAMFullDataset(train_src, config.target_size, config.overlap, config.mask_pad_class)
    val_ds = SAMFullDataset(val_src, config.target_size, config.overlap, config.mask_pad_class)

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
