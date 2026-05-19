from typing import Callable

import lightning as L
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW

from pato.pipelines._val import sliding_window_val_step


class UNetLightning(L.LightningModule):
    """LightningModule for UNet-shaped pipelines.

    Takes a pre-built `nn.Module` (image → logits) and a scheduler partial
    via DI. The Hydra `net:` group instantiates the model; the `lr:` group
    instantiates the scheduler factory; the training script wires them in.

    **Training** runs on the tile cache. **Validation** runs sliding-window
    inference at the cache's native `target_size` on full images — see
    `pato.pipelines._val.sliding_window_val_step`. Only `val_dice` is
    logged (no `val_loss`); checkpoint selection is on `val_dice` (max).
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1.0e-4,
        scheduler_partial: Callable | None = None,
        val_target_size: int = 512,
        val_overlap: int = 64,
    ):
        super().__init__()
        # `model` and `scheduler_partial` are not yaml-serializable — they
        # are runtime-injected; hparams stores only the scalars.
        self.save_hyperparameters(ignore=["model", "scheduler_partial"])
        self.model = model
        self.scheduler_partial = scheduler_partial
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_dice = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # `batch_size=1` for full-image val: `image (1,3,H,W)`, `mask (1,H,W)`.
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
        opt = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        if self.scheduler_partial is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": self.scheduler_partial(opt)}

    def to_inference_model(self) -> nn.Module:
        """Return the underlying network for use with `pato.inference.Predictor`."""
        return self.model
