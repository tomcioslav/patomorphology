"""LightningModule for the nnUnet pipeline.

Two ways this differs from `UNetLightning`:

- **Optimizer is SGD+Nesterov with momentum=0.99** (nnU-Net's choice),
  not AdamW. Pinned in `configure_optimizers`. The `lr:` group still
  controls the scheduler partial and the initial LR — pair with
  `lr=poly` to get the full nnU-Net recipe.
- **`training_step` handles deep supervision.** When the model is in
  training mode and `deep_supervision=True`, `DynUNet` returns a
  `(B, S+1, C, H, W)` stack. We compute the loss at each head and
  combine with the standard nnU-Net descending weights `[1/2^0, 1/2^1,
  ..., 1/2^S]` (renormalized).

Validation is the shared full-image sliding-window flow (see
`pato.pipelines._val`). `DynUNet` returns a single `(B, C, H, W)` head
in eval mode, so it works with `sliding_window_inference` unchanged.
"""

from typing import Callable

import lightning as L
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import SGD

from pato.pipelines._val import sliding_window_val_step


class NNUNetLightning(L.LightningModule):
    """nnU-Net-style training loop.

    `model` is expected to be a `DynUNet` (or any nn.Module that emits a
    deep-supervision-shaped output during training and a plain
    `(B, C, H, W)` tensor in eval mode). Other architectures work too —
    if their `forward` always returns `(B, C, H, W)` the deep-supervision
    branch is silently skipped.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1.0e-2,
        scheduler_partial: Callable | None = None,
        val_target_size: int = 512,
        val_overlap: int = 64,
        momentum: float = 0.99,
        weight_decay: float = 3.0e-5,
    ):
        super().__init__()
        # `model` and `scheduler_partial` are runtime-injected, not
        # yaml-serializable — hparams keeps only the scalars.
        self.save_hyperparameters(ignore=["model", "scheduler_partial"])
        self.model = model
        self.scheduler_partial = scheduler_partial
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_dice = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _multi_scale_loss(
        self, logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Weighted sum of per-head losses for deep supervision.

        `logits` shape: `(B, S+1, C, H, W)` — leading axis stacks the main
        output and `S` aux outputs (all at full resolution; MONAI's
        `DynUNet` upsamples each one internally).

        nnU-Net weights are `[1, 1/2, 1/4, ..., 1/2^S]`, renormalized to
        sum to 1. The main head (index 0) gets the largest weight.
        """
        n_heads = logits.shape[1]
        weights = torch.tensor(
            [1.0 / (2**i) for i in range(n_heads)],
            device=logits.device,
            dtype=logits.dtype,
        )
        weights = weights / weights.sum()

        masks_b1hw = masks.unsqueeze(1)
        loss = logits.new_zeros(())
        for i, w in enumerate(weights):
            loss = loss + w * self.loss_fn(logits[:, i], masks_b1hw)
        return loss

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        if logits.ndim == 5:
            # Deep supervision active.
            loss = self._multi_scale_loss(logits, masks)
        else:
            # Single-head output (e.g. deep_supervision=False, or a
            # different model class wired through this pipeline).
            loss = self.loss_fn(logits, masks.unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Val is full-image sliding window. DynUNet in eval mode returns
        # the single main head — works with sliding_window_inference.
        image, mask = batch
        sliding_window_val_step(
            model=self.model,
            image=image,
            mask=mask,
            target_size=self.hparams.val_target_size,
            overlap=self.hparams.val_overlap,
            dice_metric=self.val_dice,
            num_classes=self.model.num_classes,
        )

    def on_validation_epoch_end(self):
        dice = self.val_dice.aggregate().item()
        self.log("val_dice", dice, prog_bar=True)
        self.val_dice.reset()

    def configure_optimizers(self):
        opt = SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            nesterov=True,
            weight_decay=self.hparams.weight_decay,
        )
        if self.scheduler_partial is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": self.scheduler_partial(opt)}

    def to_inference_model(self) -> nn.Module:
        """Return the underlying network for use with `pato.inference.Predictor`."""
        return self.model
