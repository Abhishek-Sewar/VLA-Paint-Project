"""Panel geometry: a grid of surface cells that paint accumulates on.

Coordinates
-----------
u, v : surface coordinates in metres (u = arc length across, v = up). Centre is (0, 0).
3D   : panel faces +z. A viewer at +z looking toward -z sees +u on the right, +v up.

A curved panel is a cylindrical bend around the vertical axis (like a car door),
convex toward the robot, so its left and right edges recede away from the gun.
"""
from __future__ import annotations

import numpy as np


class Panel:
    def __init__(self, width: float = 1.0, height: float = 0.6,
                 resolution: float = 0.005, curvature: float = 0.0):
        if curvature < 0:
            raise ValueError("curvature must be >= 0 (1/radius, in 1/m)")
        self.width, self.height = float(width), float(height)
        self.res = float(resolution)
        self.curvature = float(curvature)
        self.nx = int(round(width / resolution))
        self.ny = int(round(height / resolution))
        self.cell_area = self.res ** 2

        self.u = (np.arange(self.nx) + 0.5) * self.res - self.width / 2
        self.v = (np.arange(self.ny) + 0.5) * self.res - self.height / 2
        U, V = np.meshgrid(self.u, self.v)          # (ny, nx)
        self.U, self.V = U, V
        self.points, self.normals = self.surface(U, V)

        self.paintable = np.ones((self.ny, self.nx), dtype=bool)
        self.cutouts: list[tuple[float, float, float, float]] = []

    # ------------------------------------------------------------------ geometry
    def surface(self, U, V):
        """Map surface coords (u, v) -> 3D points and unit normals. Works on arrays."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        if self.curvature == 0.0:
            P = np.stack([U, V, np.zeros_like(U)], axis=-1)
            N = np.zeros_like(P)
            N[..., 2] = 1.0
            return P, N
        R = 1.0 / self.curvature
        th = U / R
        s, c = np.sin(th), np.cos(th)
        P = np.stack([R * s, V, R * c - R], axis=-1)
        N = np.stack([s, np.zeros_like(s), c], axis=-1)
        return P, N

    def x_to_u(self, x):
        """Approximate inverse of surface() for the horizontal coordinate."""
        if self.curvature == 0.0:
            return x
        R = 1.0 / self.curvature
        return R * np.arcsin(np.clip(x / R, -1.0, 1.0))

    def ray_hit(self, pos, d):
        """Where a ray first meets the (infinitely extended) surface: (u, v, t) or None."""
        pos = np.asarray(pos, dtype=float)
        d = np.asarray(d, dtype=float)
        if self.curvature == 0.0:
            if d[2] >= -1e-9:
                return None
            t = -pos[2] / d[2]
        else:
            R = 1.0 / self.curvature
            ox, oz = pos[0], pos[2] + R        # ray origin relative to cylinder axis
            A = d[0] ** 2 + d[2] ** 2
            B = 2 * (ox * d[0] + oz * d[2])
            C = ox ** 2 + oz ** 2 - R ** 2
            disc = B * B - 4 * A * C
            if A < 1e-12 or disc < 0:
                return None
            t = (-B - np.sqrt(disc)) / (2 * A)  # first (outer) intersection
        if t <= 0:
            return None
        hit = pos + t * d
        if self.curvature == 0.0:
            u = hit[0]
        else:
            R = 1.0 / self.curvature
            u = R * np.arctan2(hit[0], hit[2] + R)
        return float(u), float(hit[1]), float(t)

    def uv_to_index(self, u, v):
        ix = (np.asarray(u) + self.width / 2) / self.res - 0.5
        iy = (np.asarray(v) + self.height / 2) / self.res - 0.5
        return ix, iy

    def contains_uv(self, u, v) -> np.ndarray:
        u, v = np.asarray(u), np.asarray(v)
        return (np.abs(u) <= self.width / 2) & (np.abs(v) <= self.height / 2)

    # --------------------------------------------------------------------- masks
    def add_cutout(self, u0: float, v0: float, u1: float, v1: float) -> None:
        """Mark a rectangular no-paint zone (e.g. window glass)."""
        u0, u1 = sorted((u0, u1))
        v0, v1 = sorted((v0, v1))
        m = (self.U >= u0) & (self.U <= u1) & (self.V >= v0) & (self.V <= v1)
        self.paintable[m] = False
        self.cutouts.append((u0, v0, u1, v1))

    def in_cutout(self, u, v, margin: float = 0.0) -> np.ndarray:
        u, v = np.asarray(u), np.asarray(v)
        hit = np.zeros(np.broadcast(u, v).shape, dtype=bool)
        for u0, v0, u1, v1 in self.cutouts:
            hit |= ((u >= u0 - margin) & (u <= u1 + margin)
                    & (v >= v0 - margin) & (v <= v1 + margin))
        return hit

    def region_mask(self, region: str) -> np.ndarray:
        """Geometric region from simple words: 'full', 'upper', 'lower right', ...

        This is a stand-in for the language-grounding module (a later phase replaces
        it with a VLM + segmentation model). The mask ignores cutouts; combine it
        with self.paintable to get the cells that should actually receive paint.
        """
        words = set(region.lower().replace("_", " ").replace("-", " ").split())
        m = np.ones((self.ny, self.nx), dtype=bool)
        if words & {"upper", "top"}:
            m &= self.V >= 0
        if words & {"lower", "bottom"}:
            m &= self.V < 0
        if "left" in words:
            m &= self.U < 0
        if "right" in words:
            m &= self.U >= 0
        return m
