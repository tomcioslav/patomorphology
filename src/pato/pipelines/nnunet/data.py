"""Train dataloader for the nnUnet pipeline.

Key difference from the baseline UNet pipeline: **heavy on-the-fly
augmentation**. The tile cache is identical (`nmsc-2x-unet-512`) — we just
randomly transform each tile every epoch instead of feeding it raw.

Augmentations follow nnU-Net's standard 2D recipe (minus stain-specific
augmentation, which would be a separate add-on):

- spatial: random flip (both axes), random 90° rotation, random affine
  (small rotation + scale)
- intensity: Gaussian noise, Gaussian blur, brightness/contrast/gamma
  jitter

MONAI dict transforms drive both keys with synchronized random params —
spatial transforms apply to `mask` with `nearest` interpolation, intensity
transforms only touch `image`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.utils.data
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from pato.dataset import DatasetViewer

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


def make_train_augmentations() -> Compose:
    """nnU-Net-style 2D augmentation pipeline.

    Applied to a dict `{"image": (3, H, W) float, "mask": (1, H, W) float}`
    — see `NNUNetDataset`. Probabilities are the standard nnU-Net values.
    """
    return Compose(
        [
            # ---- spatial (apply to image AND mask) ------------------------
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "mask"], prob=0.5),
            RandAffined(
                keys=["image", "mask"],
                prob=0.3,
                rotate_range=(0.52,),  # ~±30°
                scale_range=(0.2, 0.2),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            # ---- intensity (image only) -----------------------------------
            RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.15),
            RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.15),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
        ]
    )


class NNUNetDataset(torch.utils.data.Dataset):
    """Tile cache + augmentation pipeline → `(image, mask)` tensor pairs.

    Wraps a `DatasetViewer` over the tile cache. Each `__getitem__`:
      1. Loads the tile as `(image (3,H,W) float[0,1], mask (H,W) int64)`.
      2. Pushes them through `transforms` as a dict (mask carries a
         singleton channel + a temporary float cast — MONAI's spatial
         transforms expect channel-first floats throughout).
      3. Squeezes mask back to `(H, W)` int64 for the loss.

    Val and inference don't use this — they go through
    `pato.pipelines._val` with the unaugmented full-image source.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str | None = "train",
        transform: Compose | None = None,
    ):
        self._viewer = DatasetViewer(root=dataset_root, split=split)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._viewer)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._viewer[idx].to_torch()  # (3,H,W), (H,W)
        if self._transform is None:
            return image, mask
        data = {"image": image, "mask": mask.unsqueeze(0).float()}
        data = self._transform(data)
        return data["image"], data["mask"].squeeze(0).long()


def make_train_dataloader(
    dataset_root: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    augment: bool = True,
) -> torch.utils.data.DataLoader:
    """Train DataLoader over a pre-tiled cache with nnU-Net augmentation."""
    transform = make_train_augmentations() if augment else None
    train_ds = NNUNetDataset(dataset_root, split="train", transform=transform)
    return torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if num_workers > 0 else None,
    )
