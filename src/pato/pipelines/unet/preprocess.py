"""Pre-tile a normalized dataset into fixed-size tiles for UNet training.

Layout produced (identical shape to the SAM cache, just RGB instead of features):

    cache_dir/
    ├── metadata.json     # DatasetMetadata with config: {target_size, overlap, mask_pad_class}
    └── samples/
        ├── BCC_1__0000.npz   # {image: (H, W, 3) uint8, mask: (H, W) uint8}
        ├── BCC_1__0001.npz
        └── ...

Why: the on-the-fly `UNetDataset` decompresses an entire ~50 MB source `.npz`
to slice out a 512×512 tile. With a tile cache, each batch fetch reads
~750 KB / tile instead. On a CUDA box this is the difference between
"GPU at 0% util, 2 it/s" and "GPU at 70-90% util, 30+ it/s".

Splits are inherited from the upstream `DatasetViewer.metadata.splits`
and projected onto tile IDs. Idempotent: re-running skips tiles whose
`.npz` already exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from pato.dataset import DatasetViewer
from pato.schema import DatasetMetadata, PatoImage, SampleMetadata
from pato.utils.image_split import split_image


def _pad_to_min(image: np.ndarray, mask: np.ndarray, min_size: int, mask_pad_class: int):
    h, w = image.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=mask_pad_class)
    return image, mask


def preprocess(
    source: DatasetViewer,
    cache_dir: Path,
    target_size: int = 512,
    overlap: int = 64,
    mask_pad_class: int = 0,
    limit: int | None = None,
) -> Path:
    """Walk `source`, tile each image to `target_size`, write to `cache_dir`.

    `limit` truncates the source for quick smoke tests.
    """
    cache_dir = Path(cache_dir)
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    n_source = len(source) if limit is None else min(limit, len(source))
    src_meta = source.metadata

    tile_splits: dict[str, list[str]] = {k: [] for k in src_meta.splits}
    samples: dict[str, SampleMetadata] = {}

    for src_idx in tqdm(range(n_source), desc=f"tiling → {cache_dir.name}"):
        sample: PatoImage = source[src_idx]
        stem = source.sample_ids[src_idx]

        image, mask = _pad_to_min(sample.image, sample.mask, target_size, mask_pad_class)
        padded = PatoImage(image=image, mask=mask)
        tiles = split_image(padded, target_size=target_size, overlap=overlap)

        src_split = next(
            (k for k, ids in src_meta.splits.items() if stem in ids), None
        )

        for tile_idx, tile in enumerate(tiles):
            tile_id = f"{stem}__{tile_idx:04d}"
            rel_path = f"samples/{tile_id}.npz"
            out_path = cache_dir / rel_path

            if not out_path.exists():
                np.savez_compressed(
                    out_path,
                    image=tile.image,                     # (target_size, target_size, 3) uint8
                    mask=tile.mask.astype(np.uint8),       # (target_size, target_size) uint8
                )

            samples[tile_id] = SampleMetadata(
                path=rel_path,
                size=(target_size, target_size),
            )
            if src_split is not None:
                tile_splits[src_split].append(tile_id)

    metadata = DatasetMetadata(
        splits=tile_splits,
        samples=samples,
        config={
            "target_size": target_size,
            "overlap": overlap,
            "mask_pad_class": mask_pad_class,
        },
    )
    (cache_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2))
    return cache_dir
