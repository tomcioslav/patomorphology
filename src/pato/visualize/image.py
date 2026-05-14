from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

Image.MAX_IMAGE_PIXELS = None


def load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(Path(path)).convert("RGB"))


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
