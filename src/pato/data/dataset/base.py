from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from pato.schema import PatoImage

# Histopathology images can exceed Pillow's default decompression-bomb cap.
Image.MAX_IMAGE_PIXELS = None


class BaseImageMaskDataset(BaseModel, ABC):
    """Abstract dataset of paired images and segmentation masks.

    `dataset[i]` returns a `PatoImage` with:
      - image: (H, W, 3) RGB uint8
      - mask:  (H, W)    integer class IDs

    Subclasses override `_load_mask` if their masks aren't already stored
    as single-channel index images (e.g. RGB-colour-coded PNGs).
    """

    root: Path

    @property
    @abstractmethod
    def images_paths(self) -> list[Path]: ...

    @abstractmethod
    def _mask_path(self, image_path: Path) -> Path: ...

    def _load_image(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    def _load_mask(self, path: Path) -> np.ndarray:
        return np.asarray(Image.open(path))

    def __getitem__(self, index: int) -> PatoImage:
        image_path = self.images_paths[index]
        return PatoImage(
            image=self._load_image(image_path),
            mask=self._load_mask(self._mask_path(image_path)),
        )

    def __len__(self) -> int:
        return len(self.images_paths)
