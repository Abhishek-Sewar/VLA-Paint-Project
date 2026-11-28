"""Film thickness -> display colours. Plain NumPy, shared by every simulator front end
(MuJoCo texture, Isaac Sim vertex colours).

Layout convention: array row r, column c is panel cell (iy, ix) = (r, c), so row 0 is
the bottom of the panel (v = -H/2) and column 0 its left edge (u = -W/2).
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
