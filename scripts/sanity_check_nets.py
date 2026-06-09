"""Sanity-check every architecture in `conf/net/`.

For each yaml under `conf/net/`, instantiates the net via Hydra and runs
a small forward pass on a dummy input. Catches typos in `_target_`,
broken channel widths, missing kwargs, etc. — without spinning up a
full Trainer.

Usage:
    uv run python scripts/sanity_check_nets.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from pato.components.models import SAMSegHead, SAMSegmentation, UNet


def _dummy_input_for(net) -> torch.Tensor:
    """Pick the right dummy input shape for the model class."""
    if isinstance(net, UNet):
        return torch.rand(1, 3, 256, 256)
    if isinstance(net, SAMSegmentation):
        return torch.rand(1, 3, 1024, 1024)
    if isinstance(net, SAMSegHead):
        return torch.rand(1, 256, 64, 64)
    raise TypeError(f"No dummy input rule for {type(net).__name__}")


def main() -> int:
    net_dir = _PROJECT_ROOT / "conf" / "net"
    yamls = sorted(p.stem for p in net_dir.glob("*.yaml"))
    print(f"Sanity-checking {len(yamls)} net configs from {net_dir}\n")

    rows: list[tuple[str, str]] = []
    failures = 0
    with initialize_config_dir(
        config_dir=str((_PROJECT_ROOT / "conf").resolve()), version_base=None
    ):
        for name in yamls:
            try:
                cfg = compose(config_name="config", overrides=[f"net={name}"])
                net = instantiate(cfg.net).eval()
                n_params = sum(p.numel() for p in net.parameters())
                with torch.no_grad():
                    out = net(_dummy_input_for(net))
                row = (
                    name,
                    f"OK  | {type(net).__name__:18} | params={n_params:>11,} | "
                    f"output={tuple(out.shape)}",
                )
            except Exception as exc:  # noqa: BLE001 — surface anything
                failures += 1
                row = (name, f"FAIL | {type(exc).__name__}: {exc}")
            rows.append(row)
            print(f"  {row[0]:24}  {row[1]}")

    print()
    if failures:
        print(f"❌ {failures}/{len(yamls)} net configs failed")
        return 1
    print(f"✅ all {len(yamls)} net configs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
