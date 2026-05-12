"""Unit tests for the SAM-head pipeline's preprocess + migrate helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pato.schema import DatasetMetadata, SampleMetadata


def _write_sam_cache(
    root: Path,
    tile_masks: dict[str, np.ndarray],
    splits: dict[str, list[str]],
    cache_config: dict | None = None,
) -> None:
    (root / "samples").mkdir(parents=True)
    sample_meta: dict[str, SampleMetadata] = {}
    for tile_id, mask in tile_masks.items():
        features = np.zeros((256, 4, 4), dtype=np.float32)
        np.savez_compressed(
            root / "samples" / f"{tile_id}.npz",
            image=features,
            mask=mask,
        )
        sample_meta[tile_id] = SampleMetadata(
            path=f"samples/{tile_id}.npz",
            size=tuple(mask.shape),
        )
    meta = DatasetMetadata(
        splits=splits,
        samples=sample_meta,
        config=cache_config or {"mask_pad_class": 8},
    )
    (root / "metadata.json").write_text(meta.model_dump_json(indent=2))


def test_migrate_cache_drops_non_bcc_tiles_and_binarizes(tmp_path):
    from pato.dataset.convert.nmsc import NMSC_CLASSES
    from pato.pipelines.sam_head.preprocess import migrate_cache

    bcc_id = NMSC_CLASSES.index("BCC")
    scc_id = NMSC_CLASSES.index("SCC")

    cache = tmp_path / "nmsc-2x-sam-vit-base-1024"
    _write_sam_cache(
        cache,
        tile_masks={
            "BCC_1__0000": np.array([[bcc_id, 0], [0, bcc_id]], dtype=np.uint8),
            "BCC_1__0001": np.array([[0, 0], [0, bcc_id]], dtype=np.uint8),
            "SCC_1__0000": np.array([[scc_id, 0], [0, scc_id]], dtype=np.uint8),
        },
        splits={
            "train": ["BCC_1__0000", "BCC_1__0001"],
            "val": ["SCC_1__0000"],
            "test": [],
        },
        cache_config={"mask_pad_class": 8, "sam_model": "facebook/sam-vit-base"},
    )

    migrate_cache(cache)

    assert not (cache / "samples" / "SCC_1__0000.npz").exists()
    for tid, expected_sum in [("BCC_1__0000", 2), ("BCC_1__0001", 1)]:
        with np.load(cache / "samples" / f"{tid}.npz") as data:
            mask = data["mask"]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) <= {0, 1}
        assert mask.sum() == expected_sum

    meta = json.loads((cache / "metadata.json").read_text())
    assert set(meta["samples"].keys()) == {"BCC_1__0000", "BCC_1__0001"}
    assert meta["splits"]["val"] == []
    assert meta["config"]["mask_pad_class"] == 0
    assert meta["config"]["sam_model"] == "facebook/sam-vit-base"


def test_migrate_cache_is_idempotent(tmp_path):
    from pato.dataset.convert.nmsc import NMSC_CLASSES
    from pato.pipelines.sam_head.preprocess import migrate_cache

    bcc_id = NMSC_CLASSES.index("BCC")
    cache = tmp_path / "nmsc-2x-sam-vit-base-1024"
    _write_sam_cache(
        cache,
        tile_masks={"BCC_1__0000": np.array([[bcc_id, 0]], dtype=np.uint8)},
        splits={"train": ["BCC_1__0000"]},
    )

    migrate_cache(cache)

    def snapshot() -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted(cache.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(cache))] = hashlib.md5(p.read_bytes()).hexdigest()
        return out

    first = snapshot()
    migrate_cache(cache)
    second = snapshot()
    assert first == second
