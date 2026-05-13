"""Hydra entry point for training.

Each run is composed from four config groups:
    net       — architecture (e.g. unet, unet_wide, sam_head_deep, sam_full)
    dataset   — which cache to train on (e.g. nmsc-2x-unet-512)
    lr        — learning rate value + scheduler (e.g. constant_1e4, cosine)
    pipeline  — training-loop shape (unet, sam_head, sam_full)

Examples:
    uv run python scripts/train.py
    uv run python scripts/train.py net=unet_wide lr=cosine
    uv run python scripts/train.py pipeline=sam_head net=sam_head_deep \\
        dataset=nmsc-2x-sam-vit-base-1024 lr=constant_3e4

Multirun (Cartesian sweep):
    uv run python scripts/train.py -m net=unet,unet_wide,unet_narrow lr=constant_1e4,cosine
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from config import paths
from pato.pipelines.train import train


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg, runs_dir=paths.runs)


if __name__ == "__main__":
    main()
