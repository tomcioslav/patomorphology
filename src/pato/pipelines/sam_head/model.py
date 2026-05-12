import numpy as np
import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor


class SAMEncoder:
    """Frozen SAM image encoder.

    Single source of truth for the encoding step — used by `preprocess.py`
    when building the feature cache and by `predictor.py` at inference.
    Same weights, same normalization, same output → no train/infer drift.

    SAM input is fixed at 1024×1024; encoder output is `(256, 64, 64)`
    regardless of which SAM size (base/large/huge) is loaded.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str | torch.device | None = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = SamModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.processor = SamProcessor.from_pretrained(model_name)
        self.model_name = model_name

    @torch.no_grad()
    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode a single image. Accepts `(H, W, 3)` uint8.

        Returns `(256, 64, 64)` float32 features.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3); got {image.shape}")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        return embeddings.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a batch. Returns `(B, 256, 64, 64)` float32."""
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        return embeddings.cpu().numpy().astype(np.float32)


class SAMSegHead(nn.Module):
    """Tiny upsampling decoder: (B, 256, 64, 64) features → (B, num_classes, 1024, 1024).

    Four 2×-upsampling blocks (64 → 128 → 256 → 512 → 1024). Each block is a
    transposed conv + GroupNorm + GELU. Final conv emits class logits.
    Total params: ~0.3M for num_classes=12.
    """

    def __init__(
        self,
        num_classes: int = 12,
        feature_channels: int = 256,
        widths: tuple[int, ...] = (128, 64, 32, 16),
    ):
        super().__init__()
        chans = [feature_channels, *widths]
        self.up = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(chans[i], chans[i + 1], kernel_size=2, stride=2),
                    nn.GroupNorm(num_groups=8, num_channels=chans[i + 1]),
                    nn.GELU(),
                )
                for i in range(len(widths))
            ]
        )
        self.head = nn.Conv2d(chans[-1], num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.up(features))
