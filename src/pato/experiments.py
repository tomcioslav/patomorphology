"""Notebook-facing helpers for working with training runs.

Each run is a folder under `paths.runs/` that contains a `config.yaml`,
`checkpoints/`, `tensorboard/`, and (optionally) `samples/`. This module
exposes:

- `list_runs(runs_dir)` — names of all runs found
- `load_run(run_path)` — a `Run` object with config + paths
- `load_predictor(run_path)` — instantiates the right `Predictor` subclass
  for that run's pipeline (dispatched on `config.pipeline`)
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Run(BaseModel):
    """A single training run on disk."""

    name: str
    path: Path
    config: dict[str, Any]

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return self.path / "tensorboard"

    def best_checkpoint(self) -> Path | None:
        candidates = sorted(
            p for p in self.checkpoints_dir.glob("*.ckpt") if p.name != "last.ckpt"
        )
        return candidates[0] if candidates else None

    def last_checkpoint(self) -> Path:
        return self.checkpoints_dir / "last.ckpt"


def list_runs(runs_dir: Path) -> list[str]:
    if not runs_dir.exists():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if (p / "config.yaml").exists())


def list_runs_by_pipeline(runs_dir: Path, pipeline: str) -> list[str]:
    """Return run names whose `config.yaml` has the given `pipeline` field."""
    return [
        name
        for name in list_runs(runs_dir)
        if load_run(runs_dir / name).config.get("pipeline") == pipeline
    ]


def load_run(run_path: Path) -> Run:
    cfg_path = run_path / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml at {run_path}")
    return Run(
        name=run_path.name,
        path=run_path,
        config=yaml.safe_load(cfg_path.read_text()),
    )


def load_predictor(run_path: Path):
    """Instantiate the correct Predictor for the pipeline that produced this run."""
    run = load_run(run_path)
    pipeline = run.config.get("pipeline")
    if pipeline == "unet":
        from pato.pipelines.unet.predictor import UNetPredictor

        return UNetPredictor.from_run(run)
    if pipeline == "sam_head":
        from pato.pipelines.sam_head.predictor import SAMHeadPredictor

        return SAMHeadPredictor.from_run(run)
    if pipeline == "sam_full":
        from pato.pipelines.sam_full.predictor import SAMFullPredictor

        return SAMFullPredictor.from_run(run)
    raise ValueError(f"Unknown pipeline: {pipeline!r}")
