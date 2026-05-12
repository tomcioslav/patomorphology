import lightning as L
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW

from pato.pipelines.sam_full.model import SAMFullModel


class SAMFullLightning(L.LightningModule):
    """LightningModule for the full SAM-encoder + head fine-tuning pipeline.

    Two optimizer parameter groups: SAM at `sam_learning_rate` (small),
    head at `learning_rate` (larger). Loss / metric are identical to the
    other segmentation pipelines.
    """

    def __init__(
        self,
        sam_model: str = "facebook/sam-vit-base",
        num_classes: int = 2,
        feature_channels: int = 256,
        head_widths: tuple[int, ...] = (128, 64, 32, 16),
        learning_rate: float = 3e-4,
        sam_learning_rate: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SAMFullModel(
            sam_model=sam_model,
            num_classes=num_classes,
            feature_channels=feature_channels,
            head_widths=tuple(head_widths),
        )
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
            n = self.hparams.num_classes
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
        return AdamW(
            [
                {"params": self.model.sam.parameters(), "lr": self.hparams.sam_learning_rate},
                {"params": self.model.head.parameters(), "lr": self.hparams.learning_rate},
            ]
        )
