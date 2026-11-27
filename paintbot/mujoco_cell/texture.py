"""Film thickness -> RGB texels for the panel texture.

No mujoco import here, so the colour mapping can be tested on its own.

Layout: texture row r, column c holds panel cell (iy, ix) = (r, c), i.e. row 0 is
the bottom of the panel (v = -H/2) and column 0 is its left edge (u = -W/2). That
matches the mesh UVs in scene.py; the on-screen test confirms it.
"""
from __future__ import annotations

import numpy as np

BARE = np.array([0.62, 0.63, 0.65])       # bare metal
GLASS = np.array([0.45, 0.65, 0.90])
POOL = np.array([0.35, 0.02, 0.25])       # tint for > hi * target (runs / sags)


def thickness_rgb(thickness, paintable, target_um: float = 50.0,
                  paint_rgb=(0.80, 0.15, 0.10), hi: float = 1.5) -> np.ndarray:
    """(ny, nx) microns -> (ny, nx, 3) uint8."""
    th = np.asarray(thickness, dtype=np.float64)
    a = np.clip(th / target_um, 0.0, 1.0)[..., None]
    rgb = BARE * (1 - a) + np.asarray(paint_rgb) * a
    over = np.clip((th / target_um - 1.0) / (hi - 1.0), 0.0, 1.0)[..., None]
    rgb = rgb * (1 - 0.6 * over) + POOL * 0.6 * over
    rgb = np.where(np.asarray(paintable)[..., None], rgb, GLASS)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def bare_texture(panel) -> np.ndarray:
    return thickness_rgb(np.zeros((panel.ny, panel.nx)), panel.paintable)


def write_texture(model, tex_id: int, rgb: np.ndarray) -> None:
    """Copy (ny, nx, 3) uint8 into model.tex_data for texture `tex_id`."""
    adr = model.tex_adr[tex_id]
    n = model.tex_height[tex_id] * model.tex_width[tex_id] * model.tex_nchannel[tex_id]
    model.tex_data[adr:adr + n] = rgb.reshape(-1)
