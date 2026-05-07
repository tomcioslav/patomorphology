from pathlib import Path

import numpy as np
from PIL import Image

from pato.source.base import BaseImageMaskDataset

# Source-of-truth class ↔ colour mapping, taken verbatim from the dataset
# author's training code:
#   github.com/smthomas-sci/SkinCancerSegmentation/blob/master/05_patch_training.py
NMSC_CLASSES: tuple[str, ...] = (
    "EPI",  # 0  Epidermis
    "GLD",  # 1  Glands
    "INF",  # 2  Inflammation
    "RET",  # 3  Reticular dermis
    "FOL",  # 4  Hair follicles
    "PAP",  # 5  Papillary dermis
    "HYP",  # 6  Hypodermis
    "KER",  # 7  Keratin
    "BKG",  # 8  Background
    "BCC",  # 9  Basal cell carcinoma
    "SCC",  # 10 Squamous cell carcinoma
    "IEC",  # 11 Intraepidermal carcinoma
)
NMSC_CLASS_COLORS: np.ndarray = np.array(
    [
        (73, 0, 106),
        (108, 0, 115),
        (145, 1, 122),
        (181, 9, 130),
        (216, 47, 148),
        (236, 85, 157),
        (254, 246, 242),
        (248, 123, 168),
        (0, 0, 0),
        (127, 255, 255),
        (127, 255, 142),
        (255, 127, 127),
    ],
    dtype=np.uint8,
)


def _rgb_to_class_index(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB mask → (H, W) class indices."""
    # Fast path: pack RGB to 24-bit ints and exact-match the palette colours.
    packed = (
        (rgb[..., 0].astype(np.int32) << 16)
        | (rgb[..., 1].astype(np.int32) << 8)
        | rgb[..., 2].astype(np.int32)
    )
    palette_packed = (
        (palette[:, 0].astype(np.int32) << 16)
        | (palette[:, 1].astype(np.int32) << 8)
        | palette[:, 2].astype(np.int32)
    )
    out = np.full(packed.shape, 255, dtype=np.uint8)
    for class_idx, color_packed in enumerate(palette_packed):
        out[packed == color_packed] = class_idx

    # Slow path: anti-aliasing artefacts at class boundaries don't match any
    # palette colour exactly. Snap them to the nearest palette colour by L2.
    unmatched = out == 255
    if unmatched.any():
        pixels = rgb[unmatched].astype(np.int32)
        diff = pixels[:, None, :] - palette[None, :, :].astype(np.int32)
        out[unmatched] = np.argmin((diff**2).sum(axis=-1), axis=-1).astype(np.uint8)
    return out


class NMSCDataset(BaseImageMaskDataset):
    """Non-Melanoma Skin Cancer Segmentation dataset (Thomas et al., 2021).

    Expected layout under `root` (e.g. `paths.nmsc_5x`):
        Images/   *.tif (or *.tiff)
        Masks/    *.png   — RGB-encoded with `NMSC_CLASS_COLORS`,
                            same stem as the matching image
    """

    @property
    def images_paths(self) -> list[Path]:
        d = self.root / "Images"
        return sorted(list(d.glob("*.tif")) + list(d.glob("*.tiff")))

    def _mask_path(self, image_path: Path) -> Path:
        return self.root / "Masks" / f"{image_path.stem}.png"

    def _load_mask(self, path: Path) -> np.ndarray:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        return _rgb_to_class_index(rgb, NMSC_CLASS_COLORS)
