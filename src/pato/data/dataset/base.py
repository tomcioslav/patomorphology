from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

# Histopathology images can exceed Pillow's default decompression-bomb cap.
Image.MAX_IMAGE_PIXELS = None


class BaseImageMaskDataset(BaseModel, ABC):
    """Abstract dataset of paired images and segmentation masks.

    `get_image` always returns an RGB array of shape (H, W, 3).
    `get_mask`  always returns an integer array of shape (H, W) where each
    pixel is a class index. Subclasses override `_load_mask` if their masks
    are stored in a non-standard form (e.g. RGB-colour-coded PNGs).
    """

    root: Path

    @property
    @abstractmethod
    def images_paths(self) -> list[Path]: ...

    @abstractmethod
    def _mask_path(self, image_path: Path) -> Path: ...

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load a mask file as a (H, W) array of integer class IDs.

        Default assumes the file is a single-channel index image (palette
        PNG, grayscale label map). Override for datasets where masks are
        stored differently.
        """
        return np.asarray(Image.open(path))

    def get_image(self, index: int) -> np.ndarray:
        return np.asarray(Image.open(self.images_paths[index]).convert("RGB"))

    def get_mask(self, index: int) -> np.ndarray:
        mask = self._load_mask(self._mask_path(self.images_paths[index]))
        if mask.ndim != 2:
            raise ValueError(
                f"`_load_mask` must return a (H, W) integer array; got shape "
                f"{mask.shape}. Override `_load_mask` in your subclass."
            )
        return mask

    def __len__(self) -> int:
        return len(self.images_paths)
