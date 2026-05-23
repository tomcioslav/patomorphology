"""MedNeXt — ConvNeXt-style 2D segmentation network (MONAI `MedNeXt`).

A third architecture alongside `pato.components.models.unet.UNet` (MONAI
`UNet`) and `pato.components.models.dyn_unet.DynUNet` (MONAI `DynUNet`).
MedNeXt (Roy et al., *Transformer-driven Scaling of ConvNets for Medical
Image Segmentation*, MICCAI 2023) ports the ConvNeXt block — a depthwise
conv with a large kernel, then an inverted-bottleneck 1×1 expand/compress
pair — into a U-Net-shaped encoder/decoder, and scales it the way
Transformers scale (depth × width × kernel size).

Why it slots straight into `pipeline=nnunet` with no new pipeline:

- MedNeXt is designed to run *inside* the nnU-Net framework — SGD+Nesterov,
  PolyLR, deep supervision, heavy augmentation. That is exactly what
  `pipeline=nnunet` already provides, so MedNeXt is a `net=` swap only.
  Run it as `pipeline=nnunet net=mednext lr=poly`.
- `NNUNetLightning._multi_scale_loss` expects deep supervision as a single
  `(B, S+1, C, H, W)` stack with every head at full resolution — the
  contract `DynUNet` satisfies (MONAI upsamples each head internally).
  MONAI's `MedNeXt` instead returns a *tuple* of heads at *descending*
  resolutions during training. This wrapper bridges the gap: it upsamples
  every aux head back to full resolution and stacks them.

`forward` behaviour (deliberately mirrors `DynUNet`):

- `self.training=True` + `deep_supervision=True` → `(B, S+1, C, H, W)`.
  The leading axis stacks the main head and `S` aux heads, all upsampled
  to full resolution, ordered main-first / descending native resolution —
  so `NNUNetLightning`'s descending `[1, 1/2, 1/4, …]` weights line up.
- otherwise → `(B, C, H, W)`. MONAI's `MedNeXt` already returns the single
  main head in eval mode, so `pato.inference.Predictor` /
  `sliding_window_inference` work unchanged.

The default config is the MedNeXt-B preset (`conf/net/mednext.yaml`).
`kernel_size` defaults to 3; the paper's kernel-5 models need the UpKern
warm-start trick to train well, so keep it at 3 unless that is wired up.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import MedNeXt as MonaiMedNeXt


def _as_list(value: object) -> object:
    """Pass ints through; coerce any sequence (incl. Hydra `ListConfig`) to `list`."""
    return value if isinstance(value, int) else list(value)  # type: ignore[call-overload]


class MedNeXt(nn.Module):
    """2D ConvNeXt-style segmentation network (MedNeXt-B by default).

    Defaults reproduce MedNeXt-B from MONAI's `create_mednext` factory:
    4 encoder / decoder stages of 2 blocks each, the `(2, 3, 4, 4)` /
    `(4, 4, 3, 2)` expansion-ratio ladders, a 4× bottleneck, residual
    connections, GroupNorm, and `init_filters=32`. For a lighter A/B,
    set every expansion ratio to 2 (the MedNeXt-S preset).
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        init_filters: int = 32,
        kernel_size: int = 3,
        encoder_expansion_ratio: list[int] | int = (2, 3, 4, 4),
        decoder_expansion_ratio: list[int] | int = (4, 4, 3, 2),
        bottleneck_expansion_ratio: int = 4,
        blocks_down: list[int] = (2, 2, 2, 2),
        blocks_bottleneck: int = 2,
        blocks_up: list[int] = (2, 2, 2, 2),
        norm_type: str = "group",
        deep_supervision: bool = True,
        use_residual_connection: bool = True,
        global_resp_norm: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.net = MonaiMedNeXt(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            init_filters=init_filters,
            kernel_size=kernel_size,
            encoder_expansion_ratio=_as_list(encoder_expansion_ratio),
            decoder_expansion_ratio=_as_list(decoder_expansion_ratio),
            bottleneck_expansion_ratio=bottleneck_expansion_ratio,
            blocks_down=list(blocks_down),
            blocks_bottleneck=blocks_bottleneck,
            blocks_up=list(blocks_up),
            norm_type=norm_type,
            deep_supervision=deep_supervision,
            use_residual_connection=use_residual_connection,
            global_resp_norm=global_resp_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if not isinstance(out, (tuple, list)):
            # Eval mode, or deep_supervision=False — already (B, C, H, W).
            return out
        # Training + deep supervision: MONAI returns heads at descending
        # resolutions, main-first. Upsample every aux head back to the
        # main head's size and stack into the (B, S+1, C, H, W) form
        # NNUNetLightning._multi_scale_loss expects.
        main, *aux = out
        size = main.shape[-2:]
        heads = [main] + [
            F.interpolate(h, size=size, mode="bilinear", align_corners=False)
            for h in aux
        ]
        return torch.stack(heads, dim=1)
