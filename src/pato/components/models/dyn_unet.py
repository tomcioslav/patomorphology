"""nnU-Net-style 2D segmentation U-Net (MONAI `DynUNet`).

Counterpart to `pato.components.models.unet.UNet` (MONAI `UNet`). The
architectural deltas vs the baseline `UNet`:

- **Plain conv blocks (no residual units).** `num_res_units=2` in the
  baseline gives ResNet-style residual blocks; nnU-Net uses two plain
  `Conv → norm → activation` blocks per stage.
- **InstanceNorm + LeakyReLU(0.01).** The baseline `MonaiUNet` defaults to
  BatchNorm + PReLU; InstanceNorm works better at small effective batch
  sizes (we run batch_size≈4 on 512×512 tiles).
- **Deep supervision.** Auxiliary segmentation heads at intermediate
  decoder levels — the loss is a weighted sum across heads. Strong
  gradient signal for deeper layers; usually worth +1-2 Dice.
- **6 stages with nnU-Net's filter ladder.** `[32, 64, 128, 256, 320, 320]`
  — the 320 cap controls memory at the bottleneck.

`forward` behaviour with deep supervision:

- `self.training=True`  → returns `(B, S+1, C, H, W)`. The leading axis
  stacks the main output and `deep_supr_num` aux outputs, all at full
  resolution (MONAI upsamples each one internally).
- `self.training=False` → returns `(B, C, H, W)` (main head only) — so
  `pato.inference.Predictor` / `sliding_window_inference` work unchanged.

The `pato.pipelines.nnunet.module.NNUNetLightning` is what handles the
multi-scale loss in `training_step`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet as MonaiDynUNet


class DynUNet(nn.Module):
    """2D nnU-Net-style segmentation network.

    Defaults are nnU-Net's standard recipe for ~512 px 2D patches: 6
    downsampling stages, 3×3 kernels, strides `[1, 2, 2, 2, 2, 2]`,
    filter ladder `[32, 64, 128, 256, 320, 320]`, InstanceNorm +
    LeakyReLU, deep supervision on the 3 deepest decoder levels.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        kernel_size: list[int] = (3, 3, 3, 3, 3, 3),
        strides: list[int] = (1, 2, 2, 2, 2, 2),
        upsample_kernel_size: list[int] = (2, 2, 2, 2, 2),
        filters: list[int] = (32, 64, 128, 256, 320, 320),
        norm_name: str = "instance",
        deep_supervision: bool = True,
        deep_supr_num: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.net = MonaiDynUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=list(kernel_size),
            strides=list(strides),
            upsample_kernel_size=list(upsample_kernel_size),
            filters=list(filters),
            norm_name=norm_name,
            act_name=("LeakyReLU", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
