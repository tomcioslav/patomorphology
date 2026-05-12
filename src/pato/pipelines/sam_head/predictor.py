from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PIL import Image

from pato.experiments import Run
from pato.pipelines.base import BasePredictor
from pato.pipelines.sam_head.model import SAMEncoder
from pato.pipelines.sam_head.module import SAMHeadLightning
from pato.schema import DatasetMetadata, PatoImage

Image.MAX_IMAGE_PIXELS = None


class SAMHeadPredictor(BasePredictor):
    """Inference-side mirror of the SAM-head pipeline.

    Owns: image loading, SAM encoding (via the same `SAMEncoder` used at
    preprocess time — no train/infer drift), head forward, sliding-window
    stitch, argmax. Same `predict()` signature as `UNetPredictor`.
    """

    def __init__(
        self,
        encoder: SAMEncoder,
        head: SAMHeadLightning,
        target_size: int,
        overlap: int,
    ):
        self.encoder = encoder
        self.head = head.eval().to(encoder.device)
        self.target_size = target_size
        self.overlap = overlap
        self.device = encoder.device

    @classmethod
    def from_run(cls, run: Run) -> "SAMHeadPredictor":
        cache_dir = Path(run.config.get("cache_dir") or run.config["dataset_root"])
        metadata = DatasetMetadata.model_validate_json(
            (cache_dir / "metadata.json").read_text()
        )
        cache_cfg = metadata.config or {}

        ckpt = run.best_checkpoint() or run.last_checkpoint()
        head = SAMHeadLightning.load_from_checkpoint(ckpt)
        encoder = SAMEncoder(model_name=cache_cfg["sam_model"])
        return cls(
            encoder=encoder,
            head=head,
            target_size=cache_cfg["target_size"],
            overlap=cache_cfg["overlap"],
        )

    @torch.no_grad()
    def _encode_then_head(self, tiles: torch.Tensor) -> torch.Tensor:
        """Sliding-window predictor function.

        tiles: (B, 3, target_size, target_size) float in [0, 1].
        Returns: (B, num_classes, target_size, target_size) logits.
        """
        arrays = [
            (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for t in tiles
        ]
        features_np = self.encoder.encode_batch(arrays)
        features = torch.from_numpy(features_np).to(self.device)
        return self.head(features)

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
            sw_batch_size=1,
            predictor=self._encode_then_head,
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
