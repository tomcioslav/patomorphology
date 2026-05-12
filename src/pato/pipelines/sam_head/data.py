import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from pato.pipelines.sam_head.config import SAMHeadRunConfig
from pato.schema import DatasetMetadata

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


class SAMFeatureDataset(torch.utils.data.Dataset):
    """Reads the SAM feature cache produced by `preprocess.py`.

    The cache layout is identical to a normalized dataset:
        <cache_dir>/
        ├── metadata.json   (DatasetMetadata)
        └── samples/<id>.npz   {image: (256, 64, 64) float32, mask: (H, W) uint8}

    `dataset[i]` returns `(features, mask)` torch tensors directly:
        features: (256, 64, 64) float32
        mask:     (H, W)        int64

    Pass `split="train" | "val" | "test"` to filter by split (split lists
    in `metadata.json` are tile IDs, projected from the source dataset's
    image-level splits during preprocess).
    """

    def __init__(
        self,
        cache_dir: Path,
        split: str | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata.json at {self.cache_dir}. Run preprocess first."
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


def make_dataloaders(
    config: SAMHeadRunConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train/val DataLoaders over the cached SAM features. Splits come
    from the cache's manifest (which inherited them from the normalized
    dataset that fed preprocess).
    """
    train_ds = SAMFeatureDataset(config.dataset_root, split="train")
    val_ds = SAMFeatureDataset(config.dataset_root, split="val")

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if config.num_workers > 0 else None,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
