"""Unit tests for the plotly visualization helpers."""

from __future__ import annotations

import numpy as np

from pato.visualize.image import show_side_by_side


def test_side_by_side_locks_aspect_ratio_on_every_panel():
    """Image and mask panels must both lock a 1:1 pixel aspect ratio.

    Regression: masks were drawn with `go.Heatmap` (free-stretching) while
    images used `go.Image` (aspect-preserving), so a wide slide's mask got
    stretched into a tall panel — same data, wildly different shape.
    """
    rgb = np.zeros((20, 80, 3), dtype=np.uint8)  # wide RGB image
    mask = np.zeros((20, 80), dtype=np.uint8)  # mask, same spatial shape

    fig = show_side_by_side(rgb, mask, titles=["image", "mask"])

    # make_subplots numbering: subplot 1 → x/y, subplot 2 → x2/y2.
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1
    assert fig.layout.yaxis2.scaleanchor == "x2"
    assert fig.layout.yaxis2.scaleratio == 1
