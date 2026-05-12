"""Hydra entry point for training.

Single run:
    uv run python scripts/train.py
    uv run python scripts/train.py pipeline=unet_quick
    uv run python scripts/train.py pipeline=sam_head dataset=nmsc-5x

Multirun (Cartesian sweep over the listed values):
    uv run python scripts/train.py -m pipeline=unet,unet_quick
    uv run python scripts/train.py -m pipeline.target_size=256,512 pipeline.learning_rate=1e-3,1e-4
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from config import paths
from pato.pipelines.train import train


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.data_processed_root = str(paths.data_processed)
        cfg.runs_dir = str(paths.runs)

    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    run_config = instantiate(cfg.pipeline)
    train(
        run_config,
        runs_dir=Path(cfg.runs_dir),
        fast_dev_run=cfg.fast_dev_run,
    )


if __name__ == "__main__":
    main()
