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

KEEP_SLIDE_PREFIXES: tuple[str, ...] = ("BCC_",)
"""Slide filename prefixes to keep when converting/migrating. The NMSC
dataset names files `BCC_*.tif` / `SCC_*.tif` / `IEC_*.tif` so prefix
matches dominant-cancer type. To extend (e.g. include SCC) add it here
and rerun migrations."""

POSITIVE_CLASSES: tuple[str, ...] = ("BCC",)
"""Mask classes that map to 1 in the binary mask. Everything else → 0."""


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Map a 12-class integer mask to a binary {0, 1} uint8 mask.

    Pixels whose class name is in `POSITIVE_CLASSES` become 1; everything
    else becomes 0. NOT identity on already-binary input — callers that
    care about idempotency must guard with their own pre-check.
    """
    positive_ids = [NMSC_CLASSES.index(name) for name in POSITIVE_CLASSES]
    return np.isin(mask, positive_ids).astype(np.uint8)


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
    """Convert one NMSC resolution level (e.g. `paths.nmsc_5x`) to the normalized format.

    Filters to `KEEP_SLIDE_PREFIXES` (BCC-only by default) and binarizes
    masks via `_binarize_mask`. Idempotent: a `.npz` that already exists
    on disk is not overwritten, on the assumption it was produced under
    the current filter+binarize rules. If you change those rules, delete
    the processed dir first or run `migrate_processed`.
    """
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    image_dir = raw_root / "Images"
    mask_dir = raw_root / "Masks"
    image_paths = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")))

    # Filter to slides whose stem starts with any kept prefix.
    image_paths = [
        p for p in image_paths
        if any(p.stem.startswith(prefix) for prefix in KEEP_SLIDE_PREFIXES)
    ]

    canonical = _load_split_stems(raw_root.parent)
    splits: dict[str, list[str]] = {k: [] for k in canonical}

    samples: dict[str, SampleMetadata] = {}
    for img_path in tqdm(image_paths, desc=f"converting NMSC → {out_dir.name}"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"

        image = np.asarray(Image.open(img_path).convert("RGB"))
        rgb_mask = np.asarray(Image.open(mask_path).convert("RGB"))
        mask = _binarize_mask(_rgb_to_class_index(rgb_mask))

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


def migrate_processed(processed_dir: Path) -> None:
    """In-place migration: drop non-BCC samples and binarize remaining masks.

    Idempotent. Re-running on a directory that's already been migrated
    has no effect — the slide filter has nothing to drop, and masks whose
    values already fit in `{0, 1}` are skipped by the binarize step.

    Operates on the format produced by `convert()` (a `metadata.json`
    plus `samples/<id>.npz` files).
    """
    processed_dir = Path(processed_dir)
    meta_path = processed_dir / "metadata.json"
    metadata = DatasetMetadata.model_validate_json(meta_path.read_text())

    kept: dict[str, SampleMetadata] = {}
    for sample_id, sample_meta in metadata.samples.items():
        keep = any(sample_id.startswith(prefix) for prefix in KEEP_SLIDE_PREFIXES)
        npz_path = processed_dir / sample_meta.path

        if not keep:
            if npz_path.exists():
                npz_path.unlink()
            continue

        if not npz_path.exists():
            raise FileNotFoundError(
                f"metadata.json references {sample_meta.path} but the file is missing"
            )
        with np.load(npz_path) as data:
            image = data["image"]
            mask = data["mask"]
        if set(np.unique(mask).tolist()) - {0, 1}:
            mask = _binarize_mask(mask)
            np.savez_compressed(npz_path, image=image, mask=mask)
        kept[sample_id] = sample_meta

    new_splits = {
        name: [sid for sid in ids if sid in kept]
        for name, ids in metadata.splits.items()
    }
    new_metadata = DatasetMetadata(
        splits=new_splits,
        samples=kept,
        config=metadata.config,
    )
    meta_path.write_text(new_metadata.model_dump_json(indent=2))
