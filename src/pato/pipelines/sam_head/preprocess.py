"""Build the SAM-feature cache.

Walks a `DatasetViewer`, splits each (image, mask) into `target_size`
tiles, runs each tile through the frozen SAM encoder, and writes one
`.npz` per tile under `cache_dir/samples/`. The cache layout matches a
normalized dataset on purpose:

    cache_dir/
    ├── metadata.json     # `DatasetMetadata` — name, classes, splits, samples
    └── samples/
        ├── BCC_1__0000.npz   # {image: (256, H/16, W/16) float32, mask: (H, W) uint8}
        ├── BCC_1__0001.npz
        └── ...

The `image` key in each `.npz` holds the SAM feature map (the "input
that goes into the net" for this pipeline). The training-side reader is
`SAMFeatureDataset` in `data.py`.

Splits are inherited from the upstream `DatasetViewer.metadata.splits`
and projected onto tile IDs (one source image expands to N tiles, all in
the same split as their source).

Run once per `(dataset × encoder × tile config)` combination. Resumable:
re-running skips tiles whose `.npz` already exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from pato.dataset import DatasetViewer
from pato.dataset.convert.nmsc import KEEP_SLIDE_PREFIXES, _binarize_mask
from pato.pipelines.sam_head.model import SAMEncoder
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
    target_size: int = 1024,
    overlap: int = 64,
    sam_model: str = "facebook/sam-vit-base",
    mask_pad_class: int = 0,
    limit: int | None = None,
) -> Path:
    """Build the cache. `limit` truncates the source for quick tests.

    Returns the cache_dir path. Writes a `metadata.json` conformant with
    `DatasetMetadata`, sharing the same structure as a normalized dataset.

    `mask_pad_class=0` reflects the binary label space (0 = no cancer / pad,
    1 = BCC). The default was 8 (BKG) under the 12-class label space — see
    `migrate_cache` for upgrading an old cache.
    """
    cache_dir = Path(cache_dir)
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    encoder = SAMEncoder(model_name=sam_model)

    n_source = len(source) if limit is None else min(limit, len(source))
    src_meta = source.metadata

    tile_splits: dict[str, list[str]] = {k: [] for k in src_meta.splits}
    samples: dict[str, SampleMetadata] = {}

    for src_idx in tqdm(range(n_source), desc=f"encoding → {cache_dir.name}"):
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
                features = encoder.encode(tile.image)
                np.savez_compressed(
                    out_path,
                    image=features,                          # (256, 64, 64) float32
                    mask=tile.mask.astype(np.uint8),         # (target_size, target_size)
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
            "sam_model": sam_model,
            "target_size": target_size,
            "overlap": overlap,
            "mask_pad_class": mask_pad_class,
        },
    )
    (cache_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2))
    return cache_dir


def migrate_cache(cache_dir: Path) -> None:
    """In-place migration: drop non-BCC tile samples + binarize remaining masks.

    The cache stores one `.npz` per tile under `samples/<slide-stem>__<tile-idx>.npz`.
    Tile IDs inherit the slide stem before `__`, so the slide-prefix filter
    from `pato.dataset.convert.nmsc.KEEP_SLIDE_PREFIXES` applies directly.

    Also rewrites `metadata.config.mask_pad_class` to 0 — under the binary
    label space, BKG (the old pad value, 8) no longer exists; 0 ("no
    cancer") is the correct pad.

    Idempotent: re-running on a migrated cache is a no-op.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "metadata.json"
    metadata = DatasetMetadata.model_validate_json(meta_path.read_text())

    kept: dict[str, SampleMetadata] = {}
    for tile_id, sample_meta in metadata.samples.items():
        # KEEP_SLIDE_PREFIXES entries look like "BCC_"; tile_id like "BCC_1__0000",
        # so tile_id.startswith("BCC_") is the right per-tile filter.
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
