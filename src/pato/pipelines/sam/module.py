from typing import Callable

import lightning as L
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW

from pato.components.models import SAMSegmentation


class SAMLightning(L.LightningModule):
    """Unified LightningModule for SAM-based segmentation.

    `self.model` is **always** a full `SAMSegmentation` (encoder + head).
    The `sam_frozen` flag selects the training regime:

    - `sam_frozen=True`  — head-only training on a SAM-feature cache. The
      encoder is frozen and never *called* in `forward` (the dataloader
      yields pre-computed features), but it stays a registered submodule
      so every checkpoint has the same `model.encoder.* + model.head.*`
      key structure.
    - `sam_frozen=False` — end-to-end training on a raw 1024-tile cache.
      Two optimizer groups: encoder at `sam_learning_rate`, head at
      `learning_rate`.

    Because the module structure is identical either way, warm-starting an
    unfrozen run from a frozen run's checkpoint (or vice versa) is a plain
    `state_dict` load — no key remapping.
    """

    def __init__(
        self,
        model: SAMSegmentation,
        sam_frozen: bool = True,
        learning_rate: float = 3.0e-4,
        sam_learning_rate: float = 1.0e-6,
        scheduler_partial: Callable | None = None,
    ):
        super().__init__()
        # `model` and `scheduler_partial` are runtime-injected, not
        # yaml-serializable — hparams keeps only the scalars + the flag.
        self.save_hyperparameters(ignore=["model", "scheduler_partial"])
        self.model = model
        self.scheduler_partial = scheduler_partial
        if sam_frozen:
            # requires_grad=False on the encoder. It's never called in the
            # frozen forward path, so its train/eval mode is irrelevant.
            self.model.encoder.freeze()
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_dice = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # frozen:   x is cached features (B, 256, 64, 64) → head only
        # unfrozen: x is raw images     (B, 3, 1024, 1024) → encoder + head
        if self.hparams.sam_frozen:
            return self.model.head(x)
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        inputs, masks = batch
        logits = self(inputs)
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
        if self.hparams.sam_frozen:
            opt = AdamW(self.model.head.parameters(), lr=self.hparams.learning_rate)
        else:
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
        """The inference model is always the full `SAMSegmentation` — the
        encoder weights are in the checkpoint regardless of `sam_frozen`.
        """
        return self.model
