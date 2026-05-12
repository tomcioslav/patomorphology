import lightning as L
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW

from pato.pipelines.sam_head.model import SAMSegHead


class SAMHeadLightning(L.LightningModule):
    """LightningModule for the head only — SAM encoder is frozen and lives outside.

    Training input is `(features, mask)` from the cached `.npz` files, not raw
    images. At inference, `predictor.py` runs the encoder live and feeds
    features into this same head.
    """

    def __init__(
        self,
        num_classes: int = 12,
        feature_channels: int = 256,
        head_widths: tuple[int, ...] = (128, 64, 32, 16),
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SAMSegHead(
            num_classes=num_classes,
            feature_channels=feature_channels,
            widths=tuple(head_widths),
        )
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
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)
