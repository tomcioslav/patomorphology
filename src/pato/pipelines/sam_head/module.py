from typing import Callable

import lightning as L
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW

from pato.components.models import SAMImageEncoder, SAMSegHead, SAMSegmentation


class SAMHeadLightning(L.LightningModule):
    """LightningModule for `sam_head` — trains only the head on cached features.

    Takes a `SAMSegHead` and a scheduler partial via DI. `sam_model` records
    which SAM encoder produced the cache; it isn't part of the training
    graph but is used at inference time to re-attach a frozen encoder
    (`to_inference_model()`).
    """

    def __init__(
        self,
        model: SAMSegHead,
        sam_model: str = "facebook/sam-vit-base",
        learning_rate: float = 1.0e-3,
        scheduler_partial: Callable | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "scheduler_partial"])
        self.model = model
        self.scheduler_partial = scheduler_partial
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_dice = DiceMetric(include_background=True, reduction="mean")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        features, masks = batch          # (B, 256, 64, 64), (B, 1024, 1024)
        logits = self(features)          # (B, num_classes, 1024, 1024)
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

    def to_inference_model(self) -> SAMSegmentation:
        """Reconstitute the full inference model.

        The training graph contains only the head (features are pre-cached),
        so the SAM encoder is loaded fresh from HF here, frozen, and bolted
        onto the trained head.
        """
        encoder = SAMImageEncoder(sam_model=self.hparams.sam_model).freeze()
        return SAMSegmentation(encoder=encoder, head=self.model)
