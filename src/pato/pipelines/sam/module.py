from typing import Callable

import lightning as L
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW

from pato.components.models import SAMSegmentation
from pato.pipelines._val import sliding_window_val_step


class SAMLightning(L.LightningModule):
    """Unified LightningModule for SAM-based segmentation.

    `self.model` is **always** a full `SAMSegmentation` (encoder + head).
    The `sam_frozen` flag selects the **training** regime:

    - `sam_frozen=True`  — head-only training on a SAM-feature cache. The
      encoder is frozen and never *called* in train `forward` (the
      dataloader yields pre-computed features), but it stays a registered
      submodule so every checkpoint has the same `model.encoder.* +
      model.head.*` key structure.
    - `sam_frozen=False` — end-to-end training on a raw 1024-tile cache.
      Two optimizer groups: encoder at `sam_learning_rate`, head at
      `learning_rate`.

    Validation is **identical for both regimes**: full-image sliding-window
    inference through encoder + head (see `pato.pipelines._val`). Only
    `val_dice` is logged (no `val_loss`); checkpoint selection is on
    `val_dice` (max).

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
        gradient_checkpointing: bool = False,
        scheduler_partial: Callable | None = None,
        val_target_size: int = 1024,
        val_overlap: int = 64,
    ):
        super().__init__()
        # `model` and `scheduler_partial` are runtime-injected, not
        # yaml-serializable — hparams keeps only the scalars + the flags.
        self.save_hyperparameters(ignore=["model", "scheduler_partial"])
        self.model = model
        self.scheduler_partial = scheduler_partial
        if sam_frozen:
            # requires_grad=False on the encoder. It's never called in the
            # frozen train forward path, so its train/eval mode is
            # irrelevant for training — but it IS called at val time.
            self.model.encoder.freeze()
        elif gradient_checkpointing:
            # End-to-end SAM fine-tuning: the ViT's stored activations are
            # the memory hog. Gradient checkpointing recomputes them in the
            # backward pass instead — ~20-30% slower, big activation-memory
            # cut. `use_reentrant=False` is the robust variant (doesn't need
            # inputs to require grad). Older transformers lack the kwarg.
            try:
                self.model.encoder.sam.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.encoder.sam.gradient_checkpointing_enable()
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_dice = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # frozen:   x is cached features (B, 256, 64, 64) → head only
        # unfrozen: x is raw images     (B, 3, 1024, 1024) → encoder + head
        if self.hparams.sam_frozen:
            return self.model.head(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, masks = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, masks.unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Full-image val regardless of `sam_frozen` — always run the full
        # `SAMSegmentation` (encoder + head) on raw images. `self.model`
        # is the deployed inference path; sidestep `self.forward`'s
        # frozen/unfrozen branch.
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
