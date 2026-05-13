from typing import Callable

import lightning as L
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW

from pato.components.models import SAMSegmentation


class SAMFullLightning(L.LightningModule):
    """LightningModule for `sam_full` — end-to-end SAM + head fine-tuning.

    Takes a pre-built `SAMSegmentation` (encoder + head) and a scheduler
    partial via DI. Two optimizer parameter groups: encoder at
    `sam_learning_rate` (small), head at `learning_rate` (larger). The
    scheduler, if any, is applied to that combined optimizer.
    """

    def __init__(
        self,
        model: SAMSegmentation,
        learning_rate: float = 3.0e-4,
        sam_learning_rate: float = 1.0e-6,
        scheduler_partial: Callable | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "scheduler_partial"])
        self.model = model
        self.scheduler_partial = scheduler_partial
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_dice = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, masks = batch                    # (B, 3, 1024, 1024), (B, 1024, 1024)
        logits = self(images)                    # (B, num_classes, 1024, 1024)
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
        opt = AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": self.hparams.sam_learning_rate},
                {"params": self.model.head.parameters(), "lr": self.hparams.learning_rate},
            ]
        )
        if self.scheduler_partial is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": self.scheduler_partial(opt)}

    def to_inference_model(self) -> SAMSegmentation:
        """Reconstitute the inference model — for sam_full it's already
        a `SAMSegmentation`, just return it.
        """
        return self.model
