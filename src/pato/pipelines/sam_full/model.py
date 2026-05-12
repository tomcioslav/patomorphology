import torch
import torch.nn as nn
from transformers import SamModel

from pato.pipelines.sam_head.model import SAMSegHead


class SAMFullModel(nn.Module):
    """SAM encoder + segmentation head, end-to-end trainable.

    Forward: `(B, 3, 1024, 1024) float[0,1]` → `(B, num_classes, 1024, 1024)` logits.
    Normalization (ImageNet mean/std) happens inside, so callers pass raw `[0, 1]`.
    """

    def __init__(
        self,
        sam_model: str = "facebook/sam-vit-base",
        num_classes: int = 2,
        feature_channels: int = 256,
        head_widths: tuple[int, ...] = (128, 64, 32, 16),
    ):
        super().__init__()
        self.sam = SamModel.from_pretrained(sam_model)
        self.head = SAMSegHead(
            num_classes=num_classes,
            feature_channels=feature_channels,
            widths=tuple(head_widths),
        )
        self.register_buffer(
            "pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        features = self.sam.vision_encoder(x)[0]   # (B, 256, 64, 64)
        return self.head(features)                  # (B, num_classes, 1024, 1024)
