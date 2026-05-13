"""Shared contract for dataset/cache builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DatasetBuilder(ABC):
    """Builds a `data/processed/<name>/` directory in the canonical shape.

    Output layout (same for every builder — what's inside each `.npz` varies):

        out_dir/
        ├── metadata.json       # DatasetMetadata: {splits, samples, config}
        └── samples/<id>.npz    # {image: array, mask: (H, W) uint8}

    Subclass contract:
      - Configuration goes in `__init__` (source dataset, target tile size,
        encoder model, palette, etc.).
      - `build(out_dir)` writes the dataset to `out_dir` and returns the
        path. **Idempotent** — re-running on an existing dir skips any
        sample whose `.npz` is already on disk; the `metadata.json` is
        rewritten each time so split membership stays current.
    """

    @abstractmethod
    def build(self, out_dir: Path) -> Path:
        """Write the dataset to `out_dir`. Returns `out_dir`."""
