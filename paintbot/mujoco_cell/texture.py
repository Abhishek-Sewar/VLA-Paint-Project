"""Panel texture for the MuJoCo viewer. The colour mapping lives in paint_colors.py.

Layout: texture row r, column c holds panel cell (iy, ix) = (r, c), i.e. row 0 is
the bottom of the panel (v = -H/2) and column 0 is its left edge (u = -W/2). That
matches the mesh UVs in scene.py; the on-screen test confirms it.
"""
from __future__ import annotations

import numpy as np

from ..paint_colors import BARE, GLASS, POOL, thickness_rgb  # noqa: F401  (re-exported)


def bare_texture(panel) -> np.ndarray:
    return thickness_rgb(np.zeros((panel.ny, panel.nx)), panel.paintable)


def write_texture(model, tex_id: int, rgb: np.ndarray) -> None:
    """Copy (ny, nx, 3) uint8 into model.tex_data for texture `tex_id`."""
    adr = model.tex_adr[tex_id]
    n = model.tex_height[tex_id] * model.tex_width[tex_id] * model.tex_nchannel[tex_id]
    model.tex_data[adr:adr + n] = rgb.reshape(-1)
