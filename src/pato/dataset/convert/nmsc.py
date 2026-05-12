"""Convert the raw NMSC segmentation dataset into the normalized format.

Run once per resolution level:

    from config import paths
    from pato.dataset.convert.nmsc import convert
    convert(raw_root=paths.nmsc_5x, out_dir=paths.data_processed / "nmsc-5x")

Output:
    out_dir/
    ├── metadata.json     `DatasetMetadata` — includes the canonical
                           train/validation/test splits shipped with NMSC
    └── samples/<stem>.npz   {image: (H, W, 3) uint8, mask: (H, W) uint8}

The mask is decoded from the raw RGB-colour-coded PNG to integer class IDs
during conversion, so every consumer downstream sees a clean (H, W) int mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from pato.schema import DatasetMetadata, SampleMetadata

Image.MAX_IMAGE_PIXELS = None


NMSC_CLASSES: tuple[str, ...] = (
    "EPI",
    "GLD",
    "INF",
    "RET",
    "FOL",
    "PAP",
    "HYP",
    "KER",
    "BKG",
    "BCC",
    "SCC",
    "IEC",
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


def _rgb_to_class_index(rgb: np.ndarray, palette: np.ndarray = NMSC_CLASS_COLORS) -> np.ndarray:
    """(H, W, 3) RGB mask → (H, W) class indices."""
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

    unmatched = out == 255
    if unmatched.any():
        pixels = rgb[unmatched].astype(np.int32)
        diff = pixels[:, None, :] - palette[None, :, :].astype(np.int32)
        out[unmatched] = np.argmin((diff**2).sum(axis=-1), axis=-1).astype(np.uint8)
    return out


def _load_split_stems(split_dir: Path) -> dict[str, set[str]]:
    """Read NMSC's canonical {train,validation,test}_files.txt → stem sets.

    `split_dir` is the directory holding those .txt files (e.g. `paths.nmsc`).
    """
    out: dict[str, set[str]] = {}
    for our_name, file_name in (
        ("train", "train_files.txt"),
        ("val", "validation_files.txt"),
        ("test", "test_files.txt"),
    ):
        path = split_dir / file_name
        if not path.exists():
            out[our_name] = set()
            continue
        out[our_name] = {Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()}
    return out


def convert(raw_root: Path, out_dir: Path) -> Path:
    """Convert one NMSC resolution level (e.g. `paths.nmsc_5x`) to the normalized format."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    image_dir = raw_root / "Images"
    mask_dir = raw_root / "Masks"
    image_paths = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")))

    canonical = _load_split_stems(raw_root.parent)
    splits: dict[str, list[str]] = {k: [] for k in canonical}

    samples: dict[str, SampleMetadata] = {}
    for img_path in tqdm(image_paths, desc=f"converting NMSC → {out_dir.name}"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"

        image = np.asarray(Image.open(img_path).convert("RGB"))
        rgb_mask = np.asarray(Image.open(mask_path).convert("RGB"))
        mask = _rgb_to_class_index(rgb_mask)

        out_path = samples_dir / f"{stem}.npz"
        if not out_path.exists():
            np.savez_compressed(out_path, image=image, mask=mask)

        samples[stem] = SampleMetadata(
            path=f"samples/{stem}.npz",
            size=tuple(image.shape[:2]),
        )
        for split_name, stems in canonical.items():
            if stem in stems:
                splits[split_name].append(stem)
                break

    metadata = DatasetMetadata(splits=splits, samples=samples)
    (out_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2))
    return out_dir
