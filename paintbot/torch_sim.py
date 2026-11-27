"""Batched PyTorch paint simulator: many environments painting at once, on CPU or GPU.

Same physics as spray.py (the NumPy reference), rewritten so every environment
is processed in one tensor operation per step. This is the piece that plugs into
Isaac Lab later: Isaac gives us the gun pose for every env each step, we return
film thickness, quality metrics and rewards for every env.

Each env has its own:
  * panel curvature (flat or curved door) and cutouts (paintable mask)
  * target region
  * gun parameters (cone angle, flow, optimal distance) -> domain randomization

All envs share one grid size (width, height, resolution).

Two evaluation modes:
  window_half_m=None  every step touches every cell. Exact and simple.
  window_half_m=0.2   only a square patch around where the spray lands. Much less
                      work per step; accurate as long as the patch covers the
                      footprint (tests check this).
"""
from __future__ import annotations

import math

import torch

_EPS_K = 1e-6


class BatchedPaintSim:
    def __init__(self, num_envs: int, width: float = 1.0, height: float = 0.6,
                 resolution: float = 0.005, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32, window_half_m: float | None = None):
        self.E = int(num_envs)
        self.width, self.height, self.res = float(width), float(height), float(resolution)
        self.nx = int(round(width / resolution))
        self.ny = int(round(height / resolution))
        self.cell_area = self.res ** 2
        self.device, self.dtype = torch.device(device), dtype

        f = dict(device=self.device, dtype=dtype)
        self.u = (torch.arange(self.nx, **f) + 0.5) * self.res - self.width / 2   # (nx,)
        self.v = (torch.arange(self.ny, **f) + 0.5) * self.res - self.height / 2  # (ny,)
        self.U = self.u.view(1, 1, self.nx).expand(1, self.ny, self.nx)
        self.V = self.v.view(1, self.ny, 1).expand(1, self.ny, self.nx)

        E = self.E
        self.curvature = torch.zeros(E, **f)
        self.paintable = torch.ones(E, self.ny, self.nx, dtype=torch.bool, device=self.device)
        self.target = torch.ones_like(self.paintable)
        self.thickness = torch.zeros(E, self.ny, self.nx, **f)
        self.deposited = torch.zeros(E, **f)
        self.emitted = torch.zeros(E, **f)

        # gun parameters, one value per env (defaults match SprayGun)
        self.cone_tan = torch.full((E,), math.tan(math.radians(15.0)), **f)
        self.flow_rate = torch.full((E,), 0.75, **f)
        self.optimal_distance = torch.full((E,), 0.25, **f)
        self.efficiency_width = torch.full((E,), 0.08, **f)

        self.window_half_m = window_half_m
        if window_half_m is not None:
            self.K = 2 * int(math.ceil(window_half_m / self.res)) + 1
            self._offs = torch.arange(self.K, device=self.device) - self.K // 2
        self._env_ids = torch.arange(E, device=self.device)

    # ------------------------------------------------------------- configuration
    def _ids(self, env_ids):
        if env_ids is None:
            return self._env_ids
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def set_panel(self, env_ids=None, curvature: float = 0.0, cutouts=()):
        """Set geometry for some envs. cutouts: list of (u0, v0, u1, v1) rectangles."""
        ids = self._ids(env_ids)
        self.curvature[ids] = float(curvature)
        mask = torch.ones(self.ny, self.nx, dtype=torch.bool, device=self.device)
        U, V = self.U[0], self.V[0]
        for u0, v0, u1, v1 in cutouts:
            u0, u1 = sorted((u0, u1))
            v0, v1 = sorted((v0, v1))
            mask &= ~((U >= u0) & (U <= u1) & (V >= v0) & (V <= v1))
        self.paintable[ids] = mask

    def set_target(self, env_ids, mask):
        """mask: (ny, nx) bool, or (len(env_ids), ny, nx)."""
        ids = self._ids(env_ids)
        self.target[ids] = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

    def set_gun(self, env_ids=None, *, cone_half_angle_deg=None, flow_rate=None,
                optimal_distance=None, efficiency_width=None):
        """Scalars or per-env tensors. Call with random values for domain randomization."""
        ids = self._ids(env_ids)
        f = dict(device=self.device, dtype=self.dtype)
        if cone_half_angle_deg is not None:
            ang = torch.as_tensor(cone_half_angle_deg, **f)
            self.cone_tan[ids] = torch.tan(ang * (math.pi / 180.0))
        if flow_rate is not None:
            self.flow_rate[ids] = torch.as_tensor(flow_rate, **f)
        if optimal_distance is not None:
            self.optimal_distance[ids] = torch.as_tensor(optimal_distance, **f)
        if efficiency_width is not None:
            self.efficiency_width[ids] = torch.as_tensor(efficiency_width, **f)

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self.thickness[ids] = 0.0
        self.deposited[ids] = 0.0
        self.emitted[ids] = 0.0

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def _surface(U, V, k):
        """(u, v) -> 3D points and normals for curvature k. All args broadcast together.

        Uses the k->0 limit where the panel is flat, so flat and curved envs share
        one code path with no Python branching.
        """
        shape = torch.broadcast_shapes(U.shape, V.shape, k.shape)
        U, V, k = U.expand(shape), V.expand(shape), k.expand(shape)
        curved = k > _EPS_K
        k_safe = torch.where(curved, k, torch.ones_like(k))
        th = U * k_safe
        s, c = torch.sin(th), torch.cos(th)
        x = torch.where(curved, s / k_safe, U)
        z = torch.where(curved, (c - 1.0) / k_safe, torch.zeros_like(U))
        s = torch.where(curved, s, torch.zeros_like(s))
        c = torch.where(curved, c, torch.ones_like(c))
        P = torch.stack([x, V, z], dim=-1)
        N = torch.stack([s, torch.zeros_like(s), c], dim=-1)
        return P, N

    def ray_hit(self, pos, d):
        """Where each env's spray axis meets its panel surface.

        pos, d: (E, 3). Returns u, v, t (each (E,)) and a bool `valid` (E,).
        """
        k = self.curvature
        curved = k > _EPS_K
        k_safe = torch.where(curved, k, torch.ones_like(k))
        R = 1.0 / k_safe
        px, py, pz = pos.unbind(-1)
        dx, dy, dz = d.unbind(-1)

        # flat: plane z = 0
        dz_safe = torch.where(dz < -1e-9, dz, -torch.ones_like(dz))
        t_flat = -pz / dz_safe
        ok_flat = dz < -1e-9

        # curved: cylinder around the vertical axis through (0, *, -R)
        oz = pz + R
        A = dx * dx + dz * dz
        B = 2.0 * (px * dx + oz * dz)
        C = px * px + oz * oz - R * R
        disc = B * B - 4.0 * A * C
        A_safe = torch.where(A > 1e-12, A, torch.ones_like(A))
        t_cyl = (-B - torch.sqrt(disc.clamp_min(0.0))) / (2.0 * A_safe)
        ok_cyl = (A > 1e-12) & (disc >= 0)

        t = torch.where(curved, t_cyl, t_flat)
        valid = torch.where(curved, ok_cyl, ok_flat) & (t > 0)
        hx, hy, hz = px + t * dx, py + t * dy, pz + t * dz
        u = torch.where(curved, R * torch.atan2(hx, hz + R), hx)
        return u, hy, t, valid

    def aim_on_panel(self, pos, d):
        """(E,) bool: spray axis lands on a paintable cell of the panel."""
        u, v, _, valid = self.ray_hit(pos, d)
        inside = valid & (u.abs() <= self.width / 2) & (v.abs() <= self.height / 2)
        ix = ((u + self.width / 2) / self.res - 0.5).round().long().clamp(0, self.nx - 1)
        iy = ((v + self.height / 2) / self.res - 0.5).round().long().clamp(0, self.ny - 1)
        return inside & self.paintable[self._env_ids, iy, ix]

    # ---------------------------------------------------------------- simulation
    def _deposit(self, Ug, Vg, pos, d, intensity, dt):
        """Paint deposited on the cells (Ug, Vg) this step. Ug/Vg broadcast to (E, h, w)."""
        k = self.curvature.view(-1, 1, 1)
        P, N = self._surface(Ug, Vg, k)                       # (E, h, w, 3)
        p = pos.view(-1, 1, 1, 3)
        dd = d.view(-1, 1, 1, 3)
        rel = P - p
        a = (rel * dd).sum(-1)
        r2 = ((rel * rel).sum(-1) - a * a).clamp_min(0.0)
        sigma = a.clamp_min(1e-4) * self.cone_tan.view(-1, 1, 1) * 0.5
        g = torch.exp(-r2 / (2.0 * sigma * sigma)) / (2.0 * math.pi * sigma * sigma)
        cos_inc = (-(N * dd).sum(-1)).clamp(0.0, 1.0)
        dopt = self.optimal_distance.view(-1, 1, 1)
        w = self.efficiency_width.view(-1, 1, 1)
        eta = torch.exp(-(a - dopt) ** 2 / (2.0 * w * w))
        emitted = (self.flow_rate * intensity * dt).view(-1, 1, 1)
        return emitted * g * cos_inc * eta * (a > 0).to(self.dtype)

    @torch.no_grad()
    def step(self, pos, direction, intensity, dt: float = 1.0 / 30.0):
        """Advance every env by one timestep.

        pos, direction: (E, 3) gun nozzle position and spray direction
        intensity:      (E,)   0..1 (0 = spray off)
        """
        f = dict(device=self.device, dtype=self.dtype)
        pos = torch.as_tensor(pos, **f).reshape(self.E, 3)
        d = torch.as_tensor(direction, **f).reshape(self.E, 3)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        inten = torch.as_tensor(intensity, **f).expand(self.E).clamp_min(0.0)

        if self.window_half_m is None:
            dep = self._deposit(self.U, self.V, pos, d, inten, dt)
            self.thickness += dep
            self.deposited += dep.sum(dim=(1, 2)) * self.cell_area
        else:
            u, v, _, valid = self.ray_hit(pos, d)
            cx = ((u + self.width / 2) / self.res - 0.5).round().long()
            cy = ((v + self.height / 2) / self.res - 0.5).round().long()
            ix = cx.view(-1, 1) + self._offs.view(1, -1)          # (E, K)
            iy = cy.view(-1, 1) + self._offs.view(1, -1)          # (E, K)
            in_x = (ix >= 0) & (ix < self.nx)
            in_y = (iy >= 0) & (iy < self.ny)
            ixc, iyc = ix.clamp(0, self.nx - 1), iy.clamp(0, self.ny - 1)

            Ug = self.u[ixc].view(self.E, 1, self.K)
            Vg = self.v[iyc].view(self.E, self.K, 1)
            dep = self._deposit(Ug, Vg, pos, d, inten, dt)       # (E, K, K)
            keep = in_y.view(self.E, self.K, 1) & in_x.view(self.E, 1, self.K)
            keep = keep & valid.view(-1, 1, 1)
            dep = dep * keep.to(self.dtype)

            flat = (self._env_ids.view(-1, 1, 1) * (self.ny * self.nx)
                    + iyc.view(self.E, self.K, 1) * self.nx
                    + ixc.view(self.E, 1, self.K))
            self.thickness.view(-1).index_add_(0, flat.reshape(-1), dep.reshape(-1))
            self.deposited += dep.sum(dim=(1, 2)) * self.cell_area

        self.emitted += self.flow_rate * inten * dt

    # ------------------------------------------------------------------- metrics
    @torch.no_grad()
    def metrics(self, target_thickness: float = 50.0, lo: float = 0.7, hi: float = 1.5,
                overspray_frac: float = 0.2, glass_frac: float = 0.2,
                min_coverage: float = 0.80) -> dict:
        """Per-env quality metrics, each a (E,) tensor. Matches metrics.coverage_report."""
        T = float(target_thickness)
        th = self.thickness
        tgt = self.target & self.paintable
        nontarget = self.paintable & ~self.target
        glass = ~self.paintable

        def frac(cond, region):
            n = region.sum(dim=(1, 2)).to(self.dtype)
            hits = (cond & region).sum(dim=(1, 2)).to(self.dtype)
            return torch.where(n > 0, hits / n.clamp_min(1), torch.zeros_like(n))

        n_t = tgt.sum(dim=(1, 2)).to(self.dtype).clamp_min(1)
        mean_t = (th * tgt).sum(dim=(1, 2)) / n_t
        var_t = (((th - mean_t.view(-1, 1, 1)) ** 2) * tgt).sum(dim=(1, 2)) / n_t
        coverage = frac(th >= lo * T, tgt)
        m = {
            "coverage": coverage,
            "in_spec": frac((th >= lo * T) & (th <= hi * T), tgt),
            "pooling": frac(th > hi * T, tgt),
            "mean_rel_thickness": mean_t / T,
            "uniformity_cv": var_t.sqrt() / mean_t.clamp_min(1e-9),
            "overspray_nontarget": frac(th > overspray_frac * T, nontarget),
            "glass_contamination": frac(th > glass_frac * T, glass),
            "transfer_efficiency": self.deposited / self.emitted.clamp_min(1e-12),
            "accept": coverage >= min_coverage,
        }
        return m
