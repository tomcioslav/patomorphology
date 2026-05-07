from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PIL import Image

from pato.experiments import Run
from pato.pipelines.base import BasePredictor
from pato.pipelines.unet.module import UNetLightning
from pato.schema import PatoImage

# Histopathology TIFFs at native resolution can exceed Pillow's default cap.
Image.MAX_IMAGE_PIXELS = None


class UNetPredictor(BasePredictor):
    """Inference-side mirror of the UNet pipeline.

    Owns: image loading, [0,1] normalization, sliding-window inference at
    the same `target_size`/`overlap` used during training, argmax postproc.
    """

    def __init__(
        self,
        model: UNetLightning,
        target_size: int,
        overlap: int,
        sw_batch_size: int = 4,
    ):
        self.model = model.eval()
        self.target_size = target_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.device = next(model.parameters()).device

    @classmethod
    def from_run(cls, run: Run) -> "UNetPredictor":
        ckpt = run.best_checkpoint() or run.last_checkpoint()
        model = UNetLightning.load_from_checkpoint(ckpt)
        return cls(
            model=model,
            target_size=run.config["target_size"],
            overlap=run.config["overlap"],
        )

    @torch.no_grad()
    def predict(self, image: Path | str | np.ndarray | PatoImage) -> np.ndarray:
        arr = self._to_hwc_array(image)

        chw = (arr.astype(np.float32) / 255.0).transpose(2, 0, 1)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(self.device)
        )

        # `sliding_window_inference` expects overlap as a *fraction* of roi.
        overlap_frac = self.overlap / self.target_size

        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=(self.target_size, self.target_size),
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
            overlap=overlap_frac,
            mode="gaussian",
        )
        return logits.argmax(dim=1).squeeze(0).cpu().numpy()

    @staticmethod
    def _to_hwc_array(
        image: Path | str | np.ndarray | PatoImage,
    ) -> np.ndarray:
        if isinstance(image, PatoImage):
            return image.image
        if isinstance(image, np.ndarray):
            return image
        return np.asarray(Image.open(Path(image)).convert("RGB"))
