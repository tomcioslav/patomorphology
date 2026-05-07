import numpy as np
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
