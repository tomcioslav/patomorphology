from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Histopathology TIFFs at native resolution can exceed Pillow's default
# decompression-bomb cap (~89 MP). Disable it for this project.
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
