"""Normalized full-image dataset → raw-RGB tile cache.

Pre-tiles each source image at `target_size` (with `overlap`) and writes
one `.npz` per tile so the on-the-fly tile-from-slide cost (~50 MB
decompression per tile) is paid once instead of every batch.

The output shape (raw RGB tiles + binary masks) is the same regardless of
which pipeline consumes it — used by UNet at `target_size=512` and by the
end-to-end `sam_finetune` regime at `target_size=1024`. Only
`SAMFeatureBuilder` is different (it pre-encodes through SAM), and only the
frozen `sam` regime needs that.

Example:
    from config import paths
    from pato.dataset import DatasetViewer
    from pato.dataset.builders.tiles import TileBuilder

    source = DatasetViewer(root=paths.data_processed / "nmsc-2x")
    TileBuilder(source=source, target_size=512).build(
        paths.data_processed / "nmsc-2x-unet-512"
    )
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pato.dataset import DatasetViewer
from pato.dataset.builders.base import DatasetBuilder
from pato.schema import DatasetMetadata, PatoImage, SampleMetadata
from pato.utils.image_split import split_image


class TileBuilder(DatasetBuilder):
    """Tile a normalized dataset into fixed-size raw RGB crops.

    Output `.npz` contents: `image: (target_size, target_size, 3) uint8`,
    `mask: (target_size, target_size) uint8`. Pipeline-agnostic — consumers
    are UNet (512) and sam_full (1024).
    """

    def __init__(
        self,
        source: DatasetViewer,
        target_size: int = 512,
        overlap: int = 64,
        mask_pad_class: int = 0,
        limit: int | None = None,
    ):
        self.source = source
        self.target_size = target_size
        self.overlap = overlap
        self.mask_pad_class = mask_pad_class
        self.limit = limit

    def build(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        n_source = (
            len(self.source) if self.limit is None
            else min(self.limit, len(self.source))
        )
        src_meta = self.source.metadata

        tile_splits: dict[str, list[str]] = {k: [] for k in src_meta.splits}
        samples: dict[str, SampleMetadata] = {}

        for src_idx in tqdm(range(n_source), desc=f"tiling → {out_dir.name}"):
            sample: PatoImage = self.source[src_idx]
            stem = self.source.sample_ids[src_idx]

            image, mask = _pad_to_min(
                sample.image, sample.mask, self.target_size, self.mask_pad_class
            )
            padded = PatoImage(image=image, mask=mask)
            tiles = split_image(
                padded, target_size=self.target_size, overlap=self.overlap
            )

            src_split = next(
                (k for k, ids in src_meta.splits.items() if stem in ids), None
            )

            for tile_idx, tile in enumerate(tiles):
                tile_id = f"{stem}__{tile_idx:04d}"
                rel_path = f"samples/{tile_id}.npz"
                out_path = out_dir / rel_path

                if not out_path.exists():
                    np.savez_compressed(
                        out_path,
                        image=tile.image,
                        mask=tile.mask.astype(np.uint8),
                    )

                samples[tile_id] = SampleMetadata(
                    path=rel_path,
                    size=(self.target_size, self.target_size),
                )
                if src_split is not None:
                    tile_splits[src_split].append(tile_id)

        metadata = DatasetMetadata(
            splits=tile_splits,
            samples=samples,
            config={
                "target_size": self.target_size,
                "overlap": self.overlap,
                "mask_pad_class": self.mask_pad_class,
                "source_root": os.path.relpath(self.source.root, out_dir),
            },
        )
        (out_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2))
        return out_dir


def _pad_to_min(
    image: np.ndarray, mask: np.ndarray, min_size: int, mask_pad_class: int
):
    h, w = image.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=mask_pad_class)
    return image, mask
