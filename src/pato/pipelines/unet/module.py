import lightning as L
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW

from pato.pipelines.unet.model import build_unet


class UNetLightning(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        num_res_units: int = 2,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_unet(
            num_classes=num_classes,
            channels=tuple(channels),
            num_res_units=num_res_units,
        )
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
            n = self.hparams.num_classes
            preds_oh = F.one_hot(logits.argmax(dim=1), n).permute(0, 3, 1, 2).float()  # (B, C, H, W)
            masks_oh = F.one_hot(masks, n).permute(0, 3, 1, 2).float()                  # (B, C, H, W)
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
