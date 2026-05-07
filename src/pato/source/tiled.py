import torch.utils.data
from PIL import Image

from pato.source.base import BaseImageMaskDataset
from pato.schema import PatoImage
from pato.utils.image_split import _tile_starts

# Histopathology images can exceed Pillow's default decompression-bomb cap.
Image.MAX_IMAGE_PIXELS = None


class TiledDataset(torch.utils.data.Dataset):
    """Wraps a `BaseImageMaskDataset` to expose fixed-size tiles as PatoImages.

    Tile coordinates are computed at init by reading just the image headers
    (no pixel decompression). `dataset[i]` returns a `PatoImage` for the
    i-th tile; use `PatoImage.collate` as the DataLoader's `collate_fn`.

    `source_indices` filters which source images contribute tiles — pass a
    train-set or val-set list to do an image-level split with no tile leakage.
    """

    def __init__(
        self,
        source: BaseImageMaskDataset,
        target_size: int | tuple[int, int],
        overlap: int | tuple[int, int],
        source_indices: list[int] | None = None,
    ):
        self.source = source
        th, tw = (target_size, target_size) if isinstance(target_size, int) else target_size
        oh, ow = (overlap, overlap) if isinstance(overlap, int) else overlap
        self.target_size = (th, tw)
        self.overlap = (oh, ow)

        if source_indices is None:
            source_indices = list(range(len(source)))
        self.source_indices = list(source_indices)

        index: list[tuple[int, int, int]] = []
        for src_idx in self.source_indices:
            path = source.images_paths[src_idx]
            with Image.open(path) as img:
                w, h = img.size
            for y in _tile_starts(h, th, oh):
                for x in _tile_starts(w, tw, ow):
                    index.append((src_idx, y, x))
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> PatoImage:
        src_idx, y, x = self._index[idx]
        sample = self.source[src_idx]
        th, tw = self.target_size
        return PatoImage(
            image=sample.image[y : y + th, x : x + tw],
            mask=sample.mask[y : y + th, x : x + tw],
        )
