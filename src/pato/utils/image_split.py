from pato.schema import PatoImage


def split_image(
    pato_image: PatoImage,
    target_size: int | tuple[int, int],
    overlap: int | tuple[int, int],
) -> list[PatoImage]:
    """Split a PatoImage into a grid of overlapping tiles.

    target_size and overlap are given in pixels as (H, W) — pass an int for
    square tiles or symmetric overlap. Every returned tile has the requested
    target_size; the last row/column is shifted up/left so the right and
    bottom edges of the source are always covered (this means the last row
    and column overlap a bit more than the requested amount).
    """
    th, tw = (target_size, target_size) if isinstance(target_size, int) else target_size
    oh, ow = (overlap, overlap) if isinstance(overlap, int) else overlap

    H, W = pato_image.image.shape[:2]
    if th > H or tw > W:
        raise ValueError(
            f"target_size ({th}, {tw}) is larger than image ({H}, {W})"
        )

    ys = _tile_starts(H, th, oh)
    xs = _tile_starts(W, tw, ow)

    return [
        PatoImage(
            image=pato_image.image[y : y + th, x : x + tw],
            mask=pato_image.mask[y : y + th, x : x + tw],
        )
        for y in ys
        for x in xs
    ]


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < target_size ({tile})")
    starts = list(range(0, length - tile + 1, stride))
    if not starts or starts[-1] + tile < length:
        starts.append(length - tile)
    return starts
