from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from pato.experiments import Run
from pato.schema import PatoImage


class BaseRunConfig(BaseModel):
    """Common fields every pipeline's `RunConfig` must carry.

    Subclasses (e.g. `UNetRunConfig`, `SAMHeadRunConfig`) add pipeline-specific
    fields. The shared `pato.pipelines.train.train()` reads only the fields
    declared here; everything else is consumed by the pipeline's own
    `build(config)` factory and `module.py`.
    """

    pipeline: str
    dataset_root: Path
    max_epochs: int = 5
    init_from_checkpoint: Path | None = None
    precision: str = "32-true"


class BasePredictor(ABC):
    """A predictor wraps a trained model with **all** the preprocessing and
    postprocessing needed to go from a raw input image to a class-index mask.

    Subclasses live in `pato.pipelines.<name>.predictor` and own everything
    that's pipeline-specific (normalization stats, tile size, encoder model,
    sliding-window setup, etc.). The contract here is intentionally small —
    notebooks should be able to call `predict(image)` without knowing
    anything about which pipeline produced the run.
    """

    @classmethod
    @abstractmethod
    def from_run(cls, run: Run) -> "BasePredictor": ...

    @abstractmethod
    def predict(self, image: Path | str | np.ndarray | PatoImage) -> np.ndarray:
        """Return a `(H, W)` integer class-index mask for the input image."""
