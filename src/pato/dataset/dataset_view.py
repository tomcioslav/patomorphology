from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from pato.schema import DatasetMetadata, PatoImage

_DEFAULT_DATA_PROCESSED = Path("data/processed")


class DatasetViewer(BaseModel):
    """Reads a normalized dataset under `data/processed/<name>/`.

    Layout:
        <root>/
        ├── metadata.json           # DatasetMetadata
        └── samples/<id>.npz        # {image: (H, W, 3) uint8, mask: (H, W) uint8}

    Pass `split="train"` / `"val"` / `"test"` to filter to that split's samples
    (split definitions come from `metadata.json`, not random `torch.randperm`,
    so every run that points at the same processed dir uses the same partition).

    Plain pydantic — **not** a `torch.utils.data.Dataset`. Pipelines wrap
    this in their own torch `Dataset` (e.g. `UNetDataset` in
    `pipelines/unet/data.py`) because what counts as "one training sample"
    varies per pipeline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    split: str | None = None

    _metadata: DatasetMetadata = PrivateAttr()
    _sample_ids: list[str] = PrivateAttr()

    @field_validator("root", mode="before")
    @classmethod
    def _resolve_bare_name(cls, v: str | Path) -> Path:
        """Accept a bare cache name (e.g. `nmsc-2x-unet-512`) and resolve
        it as `data/processed/<name>`. Absolute and multi-segment paths
        pass through unchanged so explicit overrides keep working.
        """
        p = Path(v)
        if not p.is_absolute() and len(p.parts) == 1:
            return _DEFAULT_DATA_PROCESSED / p
        return p

    def model_post_init(self, _) -> None:
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json at {self.root}")
        self._metadata = DatasetMetadata.model_validate_json(meta_path.read_text())

        if self.split is None:
            self._sample_ids = sorted(self._metadata.samples.keys())
        else:
            if self.split not in self._metadata.splits:
                raise ValueError(
                    f"split {self.split!r} not in available splits "
                    f"{sorted(self._metadata.splits.keys())}"
                )
            self._sample_ids = list(self._metadata.splits[self.split])

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def sample_ids(self) -> list[str]:
        return list(self._sample_ids)

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, index: int) -> PatoImage:
        sample_id = self._sample_ids[index]
        sample_meta = self._metadata.samples[sample_id]
        with np.load(self.root / sample_meta.path) as data:
            image = data["image"]
            mask = data["mask"]
        return PatoImage(image=image, mask=mask)
