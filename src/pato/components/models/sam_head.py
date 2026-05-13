"""SAM segmentation head.

The decoder that turns SAM's (256, 64, 64) image embeddings into a
1024×1024 segmentation map. Spatial depth is fixed by 64→1024 geometry
(four 2× upsamples); `blocks_per_stage > 0` adds extra Conv3×3 layers at
each upsampled resolution without changing the upsampling cadence.

Used by both `sam_head` (trained alone on cached features) and `sam_full`
(trained end-to-end alongside the SAM encoder).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SAMSegHead(nn.Module):
    """Upsampling decoder: (B, 256, 64, 64) features → (B, num_classes, 1024, 1024).

    Each stage starts with a transposed conv + GroupNorm + GELU and is
    optionally followed by `blocks_per_stage` extra Conv3×3 + GroupNorm +
    GELU blocks at the upsampled resolution. Final 1×1 conv emits class
    logits.
    """

    def __init__(
        self,
        num_classes: int = 2,
        feature_channels: int = 256,
        widths: tuple[int, ...] = (128, 64, 32, 16),
        blocks_per_stage: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        chans = [feature_channels, *widths]
        stages = []
        for i in range(len(widths)):
            layers: list[nn.Module] = [
                nn.ConvTranspose2d(chans[i], chans[i + 1], kernel_size=2, stride=2),
                nn.GroupNorm(num_groups=8, num_channels=chans[i + 1]),
                nn.GELU(),
            ]
            for _ in range(blocks_per_stage):
                layers += [
                    nn.Conv2d(chans[i + 1], chans[i + 1], kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=chans[i + 1]),
                    nn.GELU(),
                ]
            stages.append(nn.Sequential(*layers))
        self.up = nn.Sequential(*stages)
        self.head = nn.Conv2d(chans[-1], num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.up(features))
