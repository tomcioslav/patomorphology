from pathlib import Path
from typing import Literal

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tifffile
from PIL import Image, UnidentifiedImageError
from plotly.subplots import make_subplots

Image.MAX_IMAGE_PIXELS = None


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except UnidentifiedImageError:
        # PIL can't read BigTIFF (different magic) or many scanner TIFFs.
        # Read the first page directly — series metadata sometimes claims
        # multiple pages that aren't actually present in scanner exports.
        arr = tifffile.imread(path, key=0)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr


def remove_green(image: np.ndarray, mode: Literal["gray", "purple"] = "gray") -> np.ndarray:
    """Strip green from pixels where G > mean(R, B). gray: drop excess; purple: redistribute it to R and B."""
    r = image[..., 0].astype(np.int16)
    g = image[..., 1].astype(np.int16)
    b = image[..., 2].astype(np.int16)
    rb_mean = (r + b) // 2
    excess = np.clip(g - rb_mean, 0, None)
    new_g = g - excess
    if mode == "gray":
        new_r, new_b = r, b
    elif mode == "purple":
        new_r = np.clip(r + excess // 2, 0, 255)
        new_b = np.clip(b + excess // 2, 0, 255)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return np.stack([new_r, new_g, new_b], axis=-1).astype(image.dtype)


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    # 2D arrays are class-index masks — nearest-neighbor preserves IDs.
    # 3D arrays are RGB images — Lanczos for clean down/upscaling.
    if image.ndim == 2:
        pil = Image.fromarray(image.astype(np.int32), mode="I").resize(size, Image.NEAREST)
        return np.asarray(pil).astype(image.dtype)
    return np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS))


def show_image(image: str | Path | np.ndarray) -> go.Figure:
    if isinstance(image, (str, Path)):
        image = load_image(image)
    fig = px.imshow(image)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def show_side_by_side(
    *images: str | Path | np.ndarray,
    titles: list[str] | None = None,
    height: int = 450,
    mask_zmax: int | None = None,
) -> go.Figure:
    """Render multiple images side-by-side as plotly subplots.

    RGB arrays (ndim=3) are shown via `go.Image`; 2D arrays (masks) via
    `go.Heatmap`. Set `mask_zmax` to fix the mask color range across plots
    so the same class is colored identically (e.g. `mask_zmax=11` for NMSC).

    Every panel locks a 1:1 pixel aspect ratio: `go.Image` preserves aspect
    on its own, but `go.Heatmap` free-stretches to fill the subplot cell, so
    without an explicit `scaleanchor` a wide slide's mask renders taller and
    narrower than the matching image — same data, mismatched shape.
    """
    n = len(images)
    titles = titles or [f"image {i}" for i in range(n)]
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles, horizontal_spacing=0.02)

    for i, img in enumerate(images, start=1):
        if isinstance(img, (str, Path)):
            img = load_image(img)
        if img.ndim == 3:
            fig.add_trace(go.Image(z=img), row=1, col=i)
        else:
            fig.add_trace(
                go.Heatmap(
                    z=img,
                    colorscale="Viridis",
                    zmin=0,
                    zmax=mask_zmax if mask_zmax is not None else int(img.max()),
                    showscale=(i == n),
                ),
                row=1,
                col=i,
            )
        x_ref = "x" if i == 1 else f"x{i}"
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(
            visible=False,
            autorange="reversed",
            scaleanchor=x_ref,
            scaleratio=1,
            row=1,
            col=i,
        )

    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def show_overlays(
    image: str | Path | np.ndarray,
    *masks: np.ndarray,
    titles: list[str] | None = None,
    height: int = 450,
    opacity: float = 0.5,
    mask_zmax: int | None = None,
    include_original: bool = False,
) -> go.Figure:
    """Render the same image multiple times with a different mask overlaid each.

    The mask is drawn on top of the RGB image with the given `opacity`, so
    each panel shows where a particular labeling (ground truth, prediction,
    ...) lands relative to the underlying tissue.

    With `include_original=True`, prepends a panel showing the bare RGB
    image (no overlay) — useful for side-by-side comparison.
    """
    if isinstance(image, (str, Path)):
        image = load_image(image)
    n_masks = len(masks)
    n_panels = n_masks + (1 if include_original else 0)

    if titles is None:
        titles = (
            (["original"] if include_original else [])
            + [f"overlay {i}" for i in range(n_masks)]
        )
    elif include_original and len(titles) == n_masks:
        # Caller passed titles for masks only — prepend a sensible default.
        titles = ["original", *titles]

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=titles, horizontal_spacing=0.02)

    zmax = mask_zmax if mask_zmax is not None else max(int(m.max()) for m in masks)

    col_offset = 0
    if include_original:
        col_offset = 1
        fig.add_trace(go.Image(z=image), row=1, col=1)
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(
            visible=False,
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=1,
        )

    for i, mask in enumerate(masks):
        col = col_offset + i + 1
        fig.add_trace(go.Image(z=image), row=1, col=col)
        fig.add_trace(
            go.Heatmap(
                z=mask,
                colorscale="Viridis",
                zmin=0,
                zmax=zmax,
                opacity=opacity,
                showscale=(col == n_panels),
            ),
            row=1,
            col=col,
        )
        x_ref = "x" if col == 1 else f"x{col}"
        fig.update_xaxes(visible=False, row=1, col=col)
        fig.update_yaxes(
            visible=False,
            autorange="reversed",
            scaleanchor=x_ref,
            scaleratio=1,
            row=1,
            col=col,
        )

    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def overlay_to_rgb(
    image: np.ndarray,
    mask: np.ndarray,
    opacity: float = 0.5,
    color: tuple[int, int, int] = (253, 231, 37),
) -> np.ndarray:
    """Blend a class-index mask onto an RGB image as a single RGB ndarray.

    Non-zero mask pixels receive `color` at `opacity` alpha; zero pixels
    pass through unchanged. Defaults to viridis-yellow so the look matches
    `show_overlays(..., mask_zmax=1)`. Multi-class masks are collapsed to
    binary foreground (any class > 0) — extend with a palette argument if
    you need per-class colors.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3); got {image.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"image/mask shape mismatch: image {image.shape[:2]} vs mask {mask.shape}"
        )
    img = image.astype(np.float32)
    overlay = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    alpha = (mask > 0).astype(np.float32)[..., None] * opacity
    blended = img * (1 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_overlay_png(
    image: np.ndarray,
    mask: np.ndarray,
    path: str | Path,
    opacity: float = 0.5,
    color: tuple[int, int, int] = (253, 231, 37),
) -> Path:
    """Save `[original | overlay]` as a single PNG file.

    No kaleido / browser export — composes the two panels as an RGB
    ndarray and writes via Pillow. Creates parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_to_rgb(image, mask, opacity=opacity, color=color)
    combined = np.concatenate([image, overlay], axis=1)
    Image.fromarray(combined).save(path)
    return path
