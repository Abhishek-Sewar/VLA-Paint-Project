"""Spray-gun deposition model (the "spray sensor" Isaac Sim doesn't ship with).

Paint leaves the gun as a Gaussian cone. For each surface cell we compute:
  a       distance along the spray axis       -> footprint size, sigma = a*tan(half_angle)/2
  r       distance from the axis              -> Gaussian falloff
  cos_inc angle between spray axis and normal -> paint spreads over more area when tilted
  eta     transfer efficiency vs. distance    -> too close / too far wastes paint

This captures the effects that mattered in the original project: spray distance
changes both thickness and footprint, and a tilted gun on a curved panel lays down
an uneven coat. Thickness is in microns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panel import Panel


@dataclass
class SprayGun:
    cone_half_angle_deg: float = 15.0
    flow_rate: float = 0.75          # micron * m^2 per second at intensity 1.0
    optimal_distance: float = 0.25   # metres
    efficiency_width: float = 0.08   # metres; std of transfer efficiency vs distance

    @property
    def tan(self) -> float:
        return float(np.tan(np.deg2rad(self.cone_half_angle_deg)))

    def sigma_at(self, distance: float) -> float:
        return max(distance, 1e-4) * self.tan * 0.5

    def efficiency(self, distance):
        return np.exp(-(np.asarray(distance) - self.optimal_distance) ** 2
                      / (2 * self.efficiency_width ** 2))

    # ----------------------------------------------------------------- core step
    def deposit(self, panel: Panel, thickness: np.ndarray, pos, direction,
                intensity: float, dt: float, window: bool = True):
        """Add one timestep of paint to `thickness` in place.

        Returns (deposited, emitted) paint volume in micron*m^2 so the caller can
        track transfer efficiency. window=True only evaluates cells near the aim
        point, which is much faster than touching the whole grid.
        """
        if intensity <= 0:
            return 0.0, 0.0
        emitted = self.flow_rate * float(intensity) * dt
        pos = np.asarray(pos, dtype=float)
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)

        sl = self._window(panel, pos, d) if window else (slice(None), slice(None))
        if sl is None:
            return 0.0, emitted

        P = panel.points[sl]
        N = panel.normals[sl]
        v = P - pos
        a = v @ d
        r2 = np.maximum(np.einsum("ijk,ijk->ij", v, v) - a * a, 0.0)
        sigma = np.maximum(a, 1e-4) * self.tan * 0.5
        g = np.exp(-r2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        cos_inc = np.clip(-(N @ d), 0.0, 1.0)
        dep = emitted * g * cos_inc * self.efficiency(a)
        dep[a <= 0] = 0.0

        thickness[sl] += dep
        return float(dep.sum() * panel.cell_area), emitted

    def _window(self, panel: Panel, pos, d):
        """Grid window around where the spray axis meets the panel surface."""
        hit = panel.ray_hit(pos, d)
        if hit is None:
            return None
        u, v, t = hit
        sigma = self.sigma_at(t)
        _, n = panel.surface(u, v)
        cos_inc = max(-float(n @ d), 0.3)
        half = 3.5 * sigma / cos_inc + 2 * panel.res
        ix0, iy0 = panel.uv_to_index(u - half, v - half)
        ix1, iy1 = panel.uv_to_index(u + half, v + half)
        x0, x1 = max(int(np.floor(ix0)), 0), min(int(np.ceil(ix1)) + 1, panel.nx)
        y0, y1 = max(int(np.floor(iy0)), 0), min(int(np.ceil(iy1)) + 1, panel.ny)
        if x0 >= x1 or y0 >= y1:
            return None
        return slice(y0, y1), slice(x0, x1)
