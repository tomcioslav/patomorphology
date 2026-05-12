from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PIL import Image

from pato.experiments import Run
from pato.pipelines.base import BasePredictor
from pato.pipelines.sam_full.module import SAMFullLightning
from pato.schema import PatoImage

Image.MAX_IMAGE_PIXELS = None


class SAMFullPredictor(BasePredictor):
    """Inference-side mirror of the sam_full pipeline.

    The LightningModule already contains SAM + head, so the predictor just
    needs sliding-window stitching at the same tile size used in training.
    """

    def __init__(
        self,
        model: SAMFullLightning,
        target_size: int,
        overlap: int,
        sw_batch_size: int = 1,
    ):
        self.model = model.eval()
        self.target_size = target_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.device = next(model.parameters()).device

    @classmethod
    def from_run(cls, run: Run) -> "SAMFullPredictor":
        ckpt = run.best_checkpoint() or run.last_checkpoint()
        model = SAMFullLightning.load_from_checkpoint(ckpt)
        return cls(
            model=model,
            target_size=run.config["target_size"],
            overlap=run.config["overlap"],
        )

    @torch.no_grad()
    def predict(self, image: Path | str | np.ndarray | PatoImage) -> np.ndarray:
        arr = self._to_hwc_array(image)

        h, w = arr.shape[:2]
        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)

        chw = (arr.astype(np.float32) / 255.0).transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(self.device)

        overlap_frac = self.overlap / self.target_size

        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=(self.target_size, self.target_size),
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
            overlap=overlap_frac,
            mode="gaussian",
        )
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        return mask[:h, :w]

    @staticmethod
    def _to_hwc_array(image) -> np.ndarray:
        if isinstance(image, PatoImage):
            return image.image
        if isinstance(image, np.ndarray):
            return image
        return np.asarray(Image.open(Path(image)).convert("RGB"))
