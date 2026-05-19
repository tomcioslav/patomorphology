"""Pipeline-agnostic full-image validation.

Both pipelines train on a tile/feature *cache* but validate on **full
images** to make `val_dice` independent of tile size and comparable across
runs. The val step here mirrors `pato.inference.Predictor`: sliding-window
inference at the model's native `target_size`, Gaussian blending, argmax,
then per-image Dice.

Three pieces:

- `FullImageValDataset` — torch Dataset over a normalized full-image
  source (e.g. `data/processed/nmsc-2x/`). One item per slide. The val
  source is named explicitly in the dataset config (`cfg.dataset.val`) —
  not derived from the train cache — so every pipeline can be pointed at
  the same source and `val_dice` is identical-protocol across runs.
- `read_cache_geometry` — pulls `target_size` and `overlap` out of the
  train cache's `metadata.config` (written by
  `TileBuilder` / `SAMFeatureBuilder`).
- `sliding_window_val_step` — runs sliding-window inference for one image
  and updates a `DiceMetric`. Called from each pipeline's
  `validation_step`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

from pato.dataset import DatasetViewer
from pato.schema import DatasetMetadata

_MP_CONTEXT = "fork" if sys.platform == "darwin" else None


class FullImageValDataset(torch.utils.data.Dataset):
    """Full-image val source. `[i]` → `(image, mask)` tensors via
    `PatoImage.to_torch()`.

    Wraps a `DatasetViewer` over a normalized dataset (the same root the
    train cache was built from). Used at `batch_size=1` because slides
    are different sizes.
    """

    def __init__(self, source_root: str | Path, split: str = "val"):
        self._viewer = DatasetViewer(root=source_root, split=split)

    def __len__(self) -> int:
        return len(self._viewer)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._viewer[idx].to_torch()


def _read_metadata(cache_root: Path) -> DatasetMetadata:
    meta_path = cache_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json at {cache_root}.")
    return DatasetMetadata.model_validate_json(meta_path.read_text())


def _resolve_cache_dir(v: str | Path) -> Path:
    """Mirror DatasetViewer's bare-name resolution. Bare names like
    `nmsc-2x-unet-512` resolve to `data/processed/nmsc-2x-unet-512`;
    absolute or multi-segment paths pass through unchanged.
    """
    p = Path(v)
    if not p.is_absolute() and len(p.parts) == 1:
        return Path("data/processed") / p
    return p


def read_cache_geometry(cache_root: str | Path) -> tuple[int, int]:
    """`(target_size, overlap)` for sliding-window val, read from the
    train cache's `metadata.config`.
    """
    cache_root = _resolve_cache_dir(cache_root)
    meta = _read_metadata(cache_root)
    cfg = meta.config or {}
    if "target_size" not in cfg or "overlap" not in cfg:
        raise ValueError(
            f"{cache_root}/metadata.json is missing `target_size`/`overlap` "
            f"in `config` — was it built by TileBuilder/SAMFeatureBuilder?"
        )
    return int(cfg["target_size"]), int(cfg["overlap"])


def make_val_dataloader(
    val_source_root: str | Path,
    num_workers: int = 2,
) -> torch.utils.data.DataLoader:
    """Full-image val loader for any pipeline.

    `val_source_root` is the normalized full-image dataset (e.g.
    `nmsc-2x`) — passed in explicitly from `cfg.dataset.val`. Bare names
    resolve under `data/processed/`. `batch_size=1` because full slides
    have different shapes (no stacking).
    """
    ds = FullImageValDataset(val_source_root, split="val")
    return torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context=_MP_CONTEXT if num_workers > 0 else None,
    )


@torch.no_grad()
def sliding_window_val_step(
    model: nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    target_size: int,
    overlap: int,
    dice_metric: DiceMetric,
    num_classes: int,
    sw_batch_size: int = 1,
) -> None:
    """Sliding-window inference + Dice update for one full image.

    Mirrors `pato.inference.Predictor.predict` exactly — same `roi_size`,
    overlap fraction, Gaussian blending — so `val_dice` reflects how the
    model will actually be deployed.

    image: (1, 3, H, W) float32; mask: (1, H, W) int64.
    """
    h, w = image.shape[-2:]

    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)

    logits = sliding_window_inference(
        inputs=image,
        roi_size=(target_size, target_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap / target_size,
        mode="gaussian",
    )
    logits = logits[..., :h, :w]

    preds_oh = (
        F.one_hot(logits.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
    )
    masks_oh = F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()
    dice_metric(preds_oh, masks_oh)
