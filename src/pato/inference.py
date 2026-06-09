"""Generic segmentation inference — one `Predictor` for every pipeline.

The per-pipeline asymmetry (does the checkpoint contain a SAM encoder or not?
is the model image→logits or features→logits?) is fully absorbed by each
LightningModule's `to_inference_model()`. By the time we reach this file we
have a plain `nn.Module` that takes `(B, 3, H, W) float[0,1]` images and
returns `(B, num_classes, H, W)` logits — UNet, SAM-head, SAM-full, all the
same shape. The Predictor then just owns sliding-window stitching + argmax.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from PIL import Image

from pato.schema import PatoImage

Image.MAX_IMAGE_PIXELS = None


class Predictor:
    """Sliding-window segmentation inference for any image→logits model.

    Construct via `pato.experiments.load_predictor(run_path)` — that loads
    the checkpoint, calls the right `to_inference_model()`, and reads
    `target_size` / `overlap` from the run's cache or config.
    """

    def __init__(
        self,
        model: nn.Module,
        target_size: int,
        overlap: int,
        sw_batch_size: int = 1,
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.target_size = target_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size

    @torch.no_grad()
    def predict(self, image: Path | str | np.ndarray | PatoImage) -> np.ndarray:
        arr = _to_hwc_array(image)
        h, w = arr.shape[:2]

        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)

        chw = (arr.astype(np.float32) / 255.0).transpose(2, 0, 1)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(chw))
            .unsqueeze(0)
            .to(self.device)
        )

        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=(self.target_size, self.target_size),
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
            overlap=self.overlap / self.target_size,
            mode="gaussian",
        )
        return logits.argmax(dim=1).squeeze(0).cpu().numpy()[:h, :w]


def _to_hwc_array(image: Path | str | np.ndarray | PatoImage) -> np.ndarray:
    if isinstance(image, PatoImage):
        return image.image
    if isinstance(image, np.ndarray):
        return image
    return np.asarray(Image.open(Path(image)).convert("RGB"))
