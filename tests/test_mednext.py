"""Unit tests for the MedNeXt wrapper's `forward` contract.

The wrapper exists to make MONAI's `MedNeXt` a drop-in for the `nnunet`
pipeline. The risky bit is the deep-supervision bridge: MONAI returns a
tuple of heads at *descending* resolutions during training, and
`NNUNetLightning._multi_scale_loss` expects a single `(B, S+1, C, H, W)`
stack at full resolution. These tests pin both forward branches.
"""

from __future__ import annotations

import torch

from pato.components.models.mednext import MedNeXt

_INPUT = torch.randn(2, 3, 64, 64)


def test_eval_mode_returns_single_full_res_head():
    """Eval mode → plain `(B, C, H, W)` — works with `sliding_window_inference`."""
    net = MedNeXt(num_classes=4, deep_supervision=True).eval()
    with torch.no_grad():
        out = net(_INPUT)
    assert out.shape == (2, 4, 64, 64)


def test_training_deep_supervision_stacks_heads_at_full_res():
    """Train + deep supervision → `(B, S+1, C, H, W)`, every head full-res."""
    net = MedNeXt(num_classes=4, deep_supervision=True).train()
    out = net(_INPUT)
    assert out.ndim == 5
    batch, heads, classes, height, width = out.shape
    assert (batch, classes, height, width) == (2, 4, 64, 64)
    assert heads > 1


def test_training_without_deep_supervision_returns_single_head():
    """`deep_supervision=False` → plain `(B, C, H, W)` even while training."""
    net = MedNeXt(num_classes=4, deep_supervision=False).train()
    out = net(_INPUT)
    assert out.shape == (2, 4, 64, 64)


def test_deep_supervision_output_feeds_nnunet_multi_scale_loss():
    """The stacked output threads through `NNUNetLightning._multi_scale_loss`."""
    from pato.pipelines.nnunet.module import NNUNetLightning

    net = MedNeXt(num_classes=4, deep_supervision=True).train()
    logits = net(_INPUT)
    masks = torch.randint(0, 4, (2, 64, 64))

    module = NNUNetLightning(model=net)
    loss = module._multi_scale_loss(logits, masks)
    assert loss.ndim == 0 and torch.isfinite(loss)
