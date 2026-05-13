"""Normalized full-image dataset → SAM feature cache.

Pre-computes SAM encoder features for each tile so the SAM-head pipeline
trains only on cached `(256, 64, 64) float32` features instead of paying
the encoder cost every batch.

Example:
    from config import paths
    from pato.dataset import DatasetViewer
    from pato.dataset.builders.sam import SAMFeatureBuilder

    source = DatasetViewer(root=paths.data_processed / "nmsc-2x")
    SAMFeatureBuilder(source=source).build(
        paths.data_processed / "nmsc-2x-sam-vit-base-1024"
    )
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pato.dataset import DatasetViewer
from pato.dataset.builders.base import DatasetBuilder
from pato.dataset.builders.nmsc import KEEP_SLIDE_PREFIXES, _binarize_mask
from pato.dataset.builders.tiles import _pad_to_min
from pato.components.models import SAMEncoder
from pato.schema import DatasetMetadata, PatoImage, SampleMetadata
from pato.utils.image_split import split_image


class SAMFeatureBuilder(DatasetBuilder):
    """Pre-compute SAM features for each tile of a normalized dataset."""

    def __init__(
        self,
        source: DatasetViewer,
        target_size: int = 1024,
        overlap: int = 64,
        sam_model: str = "facebook/sam-vit-base",
        mask_pad_class: int = 0,
        limit: int | None = None,
    ):
        self.source = source
        self.target_size = target_size
        self.overlap = overlap
        self.sam_model = sam_model
        self.mask_pad_class = mask_pad_class
        self.limit = limit

    def build(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        encoder = SAMEncoder(model_name=self.sam_model)

        n_source = (
            len(self.source) if self.limit is None
            else min(self.limit, len(self.source))
        )
        src_meta = self.source.metadata

        tile_splits: dict[str, list[str]] = {k: [] for k in src_meta.splits}
        samples: dict[str, SampleMetadata] = {}

        for src_idx in tqdm(range(n_source), desc=f"encoding → {out_dir.name}"):
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
                    features = encoder.encode(tile.image)
                    np.savez_compressed(
                        out_path,
                        image=features,
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
                "sam_model": self.sam_model,
                "target_size": self.target_size,
                "overlap": self.overlap,
                "mask_pad_class": self.mask_pad_class,
                "source_root": os.path.relpath(self.source.root, out_dir),
            },
        )
        (out_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2))
        return out_dir


def migrate_cache(cache_dir: Path) -> None:
    """In-place migration: drop non-BCC tile samples + binarize remaining masks.

    Tile IDs look like `<slide-stem>__<tile-idx>`, so the slide-prefix
    filter from `KEEP_SLIDE_PREFIXES` applies directly.

    Also rewrites `metadata.config.mask_pad_class` to 0 (binary label
    space: BKG was the old pad value 8; 0 is the correct pad now).

    Idempotent: re-running on a migrated cache is a no-op.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "metadata.json"
    metadata = DatasetMetadata.model_validate_json(meta_path.read_text())

    kept: dict[str, SampleMetadata] = {}
    for tile_id, sample_meta in metadata.samples.items():
        keep = any(tile_id.startswith(prefix) for prefix in KEEP_SLIDE_PREFIXES)
        npz_path = cache_dir / sample_meta.path

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
        kept[tile_id] = sample_meta

    new_splits = {
        name: [tid for tid in ids if tid in kept]
        for name, ids in metadata.splits.items()
    }
    new_config = dict(metadata.config or {})
    new_config["mask_pad_class"] = 0

    new_metadata = DatasetMetadata(
        splits=new_splits,
        samples=kept,
        config=new_config,
    )
    meta_path.write_text(new_metadata.model_dump_json(indent=2))
