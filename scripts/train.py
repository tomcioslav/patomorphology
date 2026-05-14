"""Hydra entry point for training.

Each run is composed from four config groups:
    net       — architecture (e.g. unet, unet_wide, sam, sam_deep)
    dataset   — which cache to train on (e.g. nmsc-2x-unet-512)
    lr        — learning rate value + scheduler (e.g. constant_1e4, cosine)
    pipeline  — training-loop shape: unet, sam (frozen, head-only), or
                sam_finetune (end-to-end — same code, sam_frozen=false preset)

Examples:
    uv run python scripts/train.py
    uv run python scripts/train.py net=unet_wide lr=cosine
    uv run python scripts/train.py pipeline=sam net=sam_deep \\
        dataset=nmsc-2x-sam-vit-base-1024 lr=constant_3e4

Multirun (Cartesian sweep):
    uv run python scripts/train.py -m net=unet,unet_wide,unet_narrow lr=constant_1e4,cosine

Warm-start from a previous run (model weights only — no optimizer state):
    uv run python scripts/train.py pipeline=sam_finetune net=sam \\
        dataset=nmsc-2x-sam-full-1024 \\
        init_from_checkpoint="'runs/<head-run>/checkpoints/best-....ckpt'"
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from pato.pipelines.train import train


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    # Hydra already created this run's directory (see the `hydra.run.dir` /
    # `hydra.sweep` block in conf/config.yaml) — train() writes all its
    # artifacts into the same folder.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    train(cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()
