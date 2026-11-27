"""Raster (boustrophedon) coverage planner: the scripted expert that generates demos.

Industrial painting robots use this pattern: horizontal passes top to bottom, gun
held perpendicular to the surface at a fixed standoff, passes overlapped so the
Gaussian footprints sum to an even coat, spray off during turnarounds and over
no-paint zones. Pass speed is computed from the target film thickness, so changing
the gun or the thickness spec changes the plan automatically.

The planner works in panel (u, v) coordinates and maps to 3D at the end, so the
same code handles flat and curved panels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panel import Panel
from .spray import SprayGun


@dataclass
class Trajectory:
    t: np.ndarray          # (T,)    seconds
    pos: np.ndarray        # (T, 3)  gun nozzle position
    dir: np.ndarray        # (T, 3)  unit spray direction
    spray: np.ndarray      # (T,)    intensity 0..1
    uv: np.ndarray         # (T, 2)  aim point in panel coords

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


@dataclass
class RasterPlanner:
    gun: SprayGun
    target_thickness: float = 50.0     # microns
    standoff: float | None = None      # defaults to the gun's optimal distance
    pass_spacing_sigma: float = 1.5    # pass spacing in footprint sigmas (overlap)
    edge_inset_sigma: float = 0.5      # first/last pass inset from region edge
    overrun_sigma: float = 0.75        # keep spraying this far past the region edge
    cutout_margin_sigma: float = 0.5   # spray off this close to a cutout
    control_hz: float = 30.0

    def plan(self, panel: Panel, target_mask: np.ndarray) -> Trajectory:
        target = target_mask & panel.paintable
        if not target.any():
            raise ValueError("target region contains no paintable cells")

        standoff = self.standoff or self.gun.optimal_distance
        sigma = self.gun.sigma_at(standoff)
        spacing = self.pass_spacing_sigma * sigma
        eta = float(self.gun.efficiency(standoff))
        # steady-state film thickness of overlapped passes = flow*eta / (speed*spacing)
        speed = self.gun.flow_rate * eta / (spacing * self.target_thickness)
        step = speed / self.control_hz

        ys, xs = np.nonzero(target)
        h = panel.res / 2
        u_min, u_max = panel.u[xs.min()] - h, panel.u[xs.max()] + h
        v_min, v_max = panel.v[ys.min()] - h, panel.v[ys.max()] + h

        inset = self.edge_inset_sigma * sigma
        top, bottom = v_max - inset, v_min + inset
        span = max(top - bottom, 0.0)
        n_rows = max(1, int(np.ceil(span / spacing)) + 1)
        rows = np.linspace(top, bottom, n_rows) if n_rows > 1 else np.array([(top + bottom) / 2])

        overrun = self.overrun_sigma * sigma
        a, b = u_min - overrun, u_max + overrun

        us, vs, on = [], [], []
        for i, v in enumerate(rows):
            start, end = (a, b) if i % 2 == 0 else (b, a)
            n = max(2, int(np.ceil(abs(end - start) / step)) + 1)
            u_line = np.linspace(start, end, n)
            us.append(u_line)
            vs.append(np.full(n, v))
            on.append(np.ones(n))
            if i + 1 < len(rows):                       # turnaround, spray off
                dv = v - rows[i + 1]
                m = max(1, int(np.ceil(dv / step)))
                v_turn = np.linspace(v, rows[i + 1], m + 1)[1:-1]
                us.append(np.full(len(v_turn), end))
                vs.append(v_turn)
                on.append(np.zeros(len(v_turn)))

        u = np.concatenate(us)
        v = np.concatenate(vs)
        spray = np.concatenate(on)
        if panel.cutouts:
            spray[panel.in_cutout(u, v, self.cutout_margin_sigma * sigma)] = 0.0

        P, N = panel.surface(u, v)
        pos = P + N * standoff
        t = np.arange(len(u)) / self.control_hz
        return Trajectory(t=t, pos=pos, dir=-N, spray=spray, uv=np.stack([u, v], axis=1))


def simulate(panel: Panel, gun: SprayGun, traj: Trajectory, window: bool = True):
    """Run a trajectory through the deposition model. Returns (thickness, deposited, emitted)."""
    thickness = np.zeros((panel.ny, panel.nx))
    dt = float(traj.t[1] - traj.t[0]) if len(traj) > 1 else 1.0 / 30
    dep_total = emit_total = 0.0
    for p, d, s in zip(traj.pos, traj.dir, traj.spray):
        dep, emit = gun.deposit(panel, thickness, p, d, s, dt, window=window)
        dep_total += dep
        emit_total += emit
    return thickness, dep_total, emit_total
