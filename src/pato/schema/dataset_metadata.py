from typing import Any

from pydantic import BaseModel


class SampleMetadata(BaseModel):
    """Per-sample entry in `DatasetMetadata.samples`."""

    path: str
    size: tuple[int, int]


class DatasetMetadata(BaseModel):
    """Manifest for a processed dataset stored under `data/processed/<name>/`.

    Same schema for raw-source conversions and pipeline-specific caches.
    The `config` slot is an optional bag for pipeline-specific facts about
    how the dataset on disk was produced (e.g. for SAM-head caches:
    `{"sam_model": "...", "target_size": 1024, "overlap": 64, ...}`).
    Anything that a downstream consumer needs to faithfully reproduce the
    encoding step belongs here, on the dataset, not on the training run.
    """

    splits: dict[str, list[str]]
    samples: dict[str, SampleMetadata]
    config: dict[str, Any] | None = None
