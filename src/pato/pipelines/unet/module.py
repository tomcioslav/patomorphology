from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW


class UNetLightning(L.LightningModule):
    """LightningModule for UNet-shaped pipelines.

    Takes a pre-built `nn.Module` (image → logits) and a scheduler partial
    via DI. The Hydra `net:` group instantiates the model; the `lr:` group
    instantiates the scheduler factory; the training script wires them in.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1.0e-4,
        scheduler_partial: Callable | None = None,
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

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.unsqueeze(1))
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if stage == "val":
            n = self.model.num_classes
            preds_oh = F.one_hot(logits.argmax(dim=1), n).permute(0, 3, 1, 2).float()
            masks_oh = F.one_hot(masks, n).permute(0, 3, 1, 2).float()
            self.val_dice(preds_oh, masks_oh)

        return loss

    def on_validation_epoch_end(self):
        dice = self.val_dice.aggregate().item()
        self.log("val_dice", dice, prog_bar=True)
        self.val_dice.reset()

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        if self.scheduler_partial is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": self.scheduler_partial(opt)}

    def to_inference_model(self) -> nn.Module:
        """Return the underlying network for use with `pato.inference.Predictor`."""
        return self.model
