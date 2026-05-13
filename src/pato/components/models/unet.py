"""2D MONAI U-Net for histopathology segmentation.

Variants (channel widths / depth) live in `conf/net/unet*.yaml`. This file
just exposes the parameterized constructor — `_target_: pato.components.models.unet.UNet`.
"""

from __future__ import annotations

import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet


class UNet(nn.Module):
    """2D segmentation U-Net.

    `strides` are always 2 between every consecutive pair of `channels`
    (standard U-Net halving). To change depth, change `channels`.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        num_res_units: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.unet = MonaiUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=tuple(channels),
            strides=(2,) * (len(channels) - 1),
            num_res_units=num_res_units,
        )

    def forward(self, x):
        return self.unet(x)
