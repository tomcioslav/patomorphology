"""SAM encoders + the unified `SAMSegmentation` inference model.

Three classes here for the SAM lineage:

  - `SAMEncoder`        — numpy-in/numpy-out, used **offline** by
                          `pato.dataset.builders.sam.SAMFeatureBuilder`
                          to build the feature cache. Not part of any
                          training graph.
  - `SAMImageEncoder`   — differentiable `nn.Module`. Used **inside**
                          `SAMSegmentation`. ImageNet normalization
                          happens inside; input is raw `[0, 1]` images.
  - `SAMSegmentation`   — `encoder + head`. The unified inference model
                          for both `sam_head` (encoder frozen, weights
                          from HF) and `sam_full` (encoder trainable,
                          weights from the checkpoint).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor

from pato.components.models.sam_head import SAMSegHead


class SAMEncoder:
    """Numpy-in/numpy-out SAM image encoder for **offline** use only.

    `@torch.no_grad()`, `.eval()`, CPU/numpy round-trip. For the
    differentiable in-graph encoder, see `SAMImageEncoder` below.

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


class SAMImageEncoder(nn.Module):
    """SAM vision encoder as a differentiable `nn.Module`.

    Wraps HuggingFace `SamModel.vision_encoder`. Input is raw `[0, 1]`
    images; ImageNet normalization happens inside, so callers pass
    un-normalized tiles. Output is `(B, 256, 64, 64)` features for
    1024×1024 inputs.

    Used inside `SAMSegmentation`. In `sam_head` it is `freeze()`d after
    loading from HF. In `sam_full` it stays trainable.
    """

    def __init__(self, sam_model: str = "facebook/sam-vit-base"):
        super().__init__()
        self.sam = SamModel.from_pretrained(sam_model)
        self.model_name = sam_model
        self.register_buffer(
            "pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        return self.sam.vision_encoder(x)[0]

    def freeze(self) -> "SAMImageEncoder":
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self


class SAMSegmentation(nn.Module):
    """The unified inference model for `sam_head` and `sam_full`.

    Training-time wrappers differ (sam_head trains only the head on
    cached features; sam_full trains both end-to-end), but the inference
    graph — image in, logits out — is identical.

    Forward: `(B, 3, 1024, 1024) float[0,1]` → `(B, num_classes, 1024, 1024)`.
    """

    def __init__(self, encoder: SAMImageEncoder, head: SAMSegHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    @property
    def num_classes(self) -> int:
        return self.head.num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
