import lightning as L
import torch
from monai.losses import DiceCELoss
from torch.optim import AdamW

from pato.models import build_unet


class UNetLightning(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 12,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_unet(num_classes=num_classes)
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.unsqueeze(1))
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)
