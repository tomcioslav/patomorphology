"""Unit tests for the NMSC raw → processed builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pato.dataset.builders.nmsc import (
    KEEP_SLIDE_PREFIXES,
    NMSC_CLASSES,
    POSITIVE_CLASSES,
    _binarize_mask,
)


def test_binarize_mask_maps_bcc_to_one_others_to_zero():
    bcc_id = NMSC_CLASSES.index("BCC")
    scc_id = NMSC_CLASSES.index("SCC")
    bkg_id = NMSC_CLASSES.index("BKG")

    mask = np.array(
        [
            [bcc_id, scc_id, bkg_id],
            [0, bcc_id, 11],
        ],
        dtype=np.uint8,
    )
    out = _binarize_mask(mask)

    expected = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)


def test_binarize_mask_all_negative_classes_yields_all_zero():
    mask = np.zeros((4, 4), dtype=np.uint8)
    out = _binarize_mask(mask)
    assert out.dtype == np.uint8
    assert out.sum() == 0


def test_binarize_mask_all_bcc_yields_all_one():
    bcc_id = NMSC_CLASSES.index("BCC")
    mask = np.full((3, 3), bcc_id, dtype=np.uint8)
    out = _binarize_mask(mask)
    assert out.dtype == np.uint8
    assert (out == 1).all()


def test_keep_slide_prefixes_and_positive_classes_default_to_bcc_only():
    assert KEEP_SLIDE_PREFIXES == ("BCC_",)
    assert POSITIVE_CLASSES == ("BCC",)


def _make_fake_raw_root(tmp_path: Path) -> Path:
    """Build a minimal NMSC-shaped raw dir with 1 BCC + 1 SCC sample.

    Layout mirrors `paths.nmsc / <res>/` — Images/, Masks/, and
    train_files.txt / validation_files.txt / test_files.txt one level up.
    """
    from PIL import Image

    res_root = tmp_path / "2x"
    img_dir = res_root / "Images"
    mask_dir = res_root / "Masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Two tiny 4x4 samples. Image is solid colour; mask is two BCC-color
    # pixels for BCC_1, two SCC-color pixels for SCC_1.
    bcc_color = (127, 255, 255)  # NMSC_CLASS_COLORS[9]
    scc_color = (127, 255, 142)  # NMSC_CLASS_COLORS[10]
    bkg_color = (0, 0, 0)        # NMSC_CLASS_COLORS[8]

    for stem, fg_color in [("BCC_1", bcc_color), ("SCC_1", scc_color)]:
        rgb_image = np.full((4, 4, 3), 200, dtype=np.uint8)
        Image.fromarray(rgb_image).save(img_dir / f"{stem}.tif")

        rgb_mask = np.full((4, 4, 3), bkg_color, dtype=np.uint8)
        rgb_mask[0, 0] = fg_color
        rgb_mask[0, 1] = fg_color
        Image.fromarray(rgb_mask).save(mask_dir / f"{stem}.png")

    # canonical split files live one level above the resolution dir
    (tmp_path / "train_files.txt").write_text("BCC_1.tif\n")
    (tmp_path / "validation_files.txt").write_text("SCC_1.tif\n")
    (tmp_path / "test_files.txt").write_text("")
    return res_root


def test_convert_keeps_only_bcc_slides_and_binarizes_masks(tmp_path):
    from pato.dataset.builders.nmsc import NMSCBuilder

    raw_root = _make_fake_raw_root(tmp_path)
    out_dir = tmp_path / "processed"

    NMSCBuilder(raw_root=raw_root).build(out_dir)

    # SCC sample must not exist on disk
    assert (out_dir / "samples" / "BCC_1.npz").exists()
    assert not (out_dir / "samples" / "SCC_1.npz").exists()

    # Mask is binary {0, 1}
    with np.load(out_dir / "samples" / "BCC_1.npz") as data:
        mask = data["mask"]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() == 2  # two BCC-coloured pixels in the input mask

    # metadata.json: splits drop SCC, samples drop SCC
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert list(meta["samples"].keys()) == ["BCC_1"]
    assert meta["splits"]["train"] == ["BCC_1"]
    assert meta["splits"]["val"] == []  # SCC_1 was in val, now gone


def _write_processed_dir(
    root: Path,
    samples: dict[str, np.ndarray],
    splits: dict[str, list[str]],
) -> None:
    """Write a minimal processed dir: one .npz per sample + metadata.json."""
    from pato.schema import DatasetMetadata, SampleMetadata

    (root / "samples").mkdir(parents=True)
    sample_meta: dict[str, SampleMetadata] = {}
    for stem, mask in samples.items():
        image = np.full((*mask.shape, 3), 200, dtype=np.uint8)
        np.savez_compressed(root / "samples" / f"{stem}.npz", image=image, mask=mask)
        sample_meta[stem] = SampleMetadata(path=f"samples/{stem}.npz", size=tuple(mask.shape))
    meta = DatasetMetadata(splits=splits, samples=sample_meta)
    (root / "metadata.json").write_text(meta.model_dump_json(indent=2))


def test_migrate_processed_drops_non_bcc_and_binarizes(tmp_path):
    from pato.dataset.builders.nmsc import NMSC_CLASSES, migrate_processed

    bcc_id = NMSC_CLASSES.index("BCC")
    scc_id = NMSC_CLASSES.index("SCC")

    bcc_mask = np.array([[bcc_id, 0], [0, bcc_id]], dtype=np.uint8)
    scc_mask = np.array([[scc_id, 0], [0, scc_id]], dtype=np.uint8)

    proc = tmp_path / "nmsc-2x"
    _write_processed_dir(
        proc,
        samples={"BCC_1": bcc_mask, "SCC_1": scc_mask},
        splits={"train": ["BCC_1"], "val": ["SCC_1"], "test": []},
    )

    migrate_processed(proc)

    assert (proc / "samples" / "BCC_1.npz").exists()
    assert not (proc / "samples" / "SCC_1.npz").exists()

    with np.load(proc / "samples" / "BCC_1.npz") as data:
        mask = data["mask"]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() == 2

    meta = json.loads((proc / "metadata.json").read_text())
    assert list(meta["samples"].keys()) == ["BCC_1"]
    assert meta["splits"]["train"] == ["BCC_1"]
    assert meta["splits"]["val"] == []


def test_migrate_processed_is_idempotent(tmp_path):
    """Running migrate_processed twice must not change the result."""
    import hashlib
    from pato.dataset.builders.nmsc import NMSC_CLASSES, migrate_processed

    bcc_id = NMSC_CLASSES.index("BCC")
    bcc_mask = np.array([[bcc_id, 0], [0, bcc_id]], dtype=np.uint8)

    proc = tmp_path / "nmsc-2x"
    _write_processed_dir(proc, {"BCC_1": bcc_mask}, {"train": ["BCC_1"]})

    migrate_processed(proc)

    def snapshot() -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted(proc.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(proc))] = hashlib.md5(p.read_bytes()).hexdigest()
        return out

    first = snapshot()
    migrate_processed(proc)
    second = snapshot()

    assert first == second
