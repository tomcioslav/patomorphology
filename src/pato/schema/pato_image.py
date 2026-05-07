import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, model_validator


class PatoImage(BaseModel):
    """A pathology image paired with its segmentation mask.

    image: (H, W, 3) RGB uint8
    mask:  (H, W)    integer class IDs
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    mask: np.ndarray

    @model_validator(mode="after")
    def _check_shapes(self) -> "PatoImage":
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3); got shape {self.image.shape}"
            )
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be (H, W); got shape {self.mask.shape}")
        if self.image.shape[:2] != self.mask.shape:
            raise ValueError(
                f"image/mask spatial shapes must match; got "
                f"image {self.image.shape[:2]} and mask {self.mask.shape}"
            )
        return self

    def to_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image, mask) as PyTorch tensors in CHW convention.

        image: (3, H, W) float32 in [0, 1]
        mask:  (H, W)    int64 class indices

        Format matches what `monai`/`torch.nn` segmentation losses expect:
        the image as a normalized float CHW tensor, the mask as a long-typed
        2D class-index tensor.
        """
        image_chw = (self.image.astype(np.float32) / 255.0).transpose(2, 0, 1)
        image = torch.from_numpy(np.ascontiguousarray(image_chw))
        mask = torch.from_numpy(self.mask.astype(np.int64))
        return image, mask

    @classmethod
    def collate(
        cls, batch: list["PatoImage"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`collate_fn` for `torch.utils.data.DataLoader`.

        Stacks a list of PatoImage into batched tensors:
            images: (B, 3, H, W) float32
            masks:  (B, H, W)    int64
        """
        pairs = [p.to_torch() for p in batch]
        images = torch.stack([img for img, _ in pairs])
        masks = torch.stack([mask for _, mask in pairs])
        return images, masks
