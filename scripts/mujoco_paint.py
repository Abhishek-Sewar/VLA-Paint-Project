"""Watch the Panda paint a panel in the MuJoCo viewer, with the coat shown live.

    python scripts/mujoco_paint.py                       # flat panel, full region
    python scripts/mujoco_paint.py --panel curved
    python scripts/mujoco_paint.py --panel door --region "upper right" --speed 2
    python scripts/mujoco_paint.py --panel door --headless   # metrics only, no window

Viewer: drag to orbit, right-drag to pan, scroll to zoom, double-click to select,
Tab/Shift-Tab for the side panels. Close the window to quit.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paintbot import Panel  # noqa: E402
from paintbot.mujoco_cell import PaintCell  # noqa: E402
from paintbot.mujoco_cell.scene import CONTROL_HZ  # noqa: E402

DOOR_WINDOW = (-0.35, 0.08, 0.35, 0.26)


def make_panel(kind: str) -> Panel:
    if kind == "curved":
        return Panel(curvature=1.5)
    p = Panel()
    if kind == "door":
        p.add_cutout(*DOOR_WINDOW)
    return p


def print_summary(cell: PaintCell, wall_s: float):
    rep = cell.report()
    print(f"\n{cell.t} ticks = {cell.t / CONTROL_HZ:.1f} s simulated in {wall_s:.1f} s")
    for k, v in rep.items():
        print(f"   {k:22s} {v:.3f}" if isinstance(v, float) else f"   {k:22s} {v}")
    for k, v in cell.tracking_stats().items():
        print(f"   {k:22s} {v:.3f}")
    val = cell.validate()
    print(f"   validation: {'PASS' if val.passed else 'FAIL ' + str(val.reasons)} {val.stats}")


def spray_cone(viewer, cell: PaintCell, rgba=(1.0, 0.75, 0.3, 0.35), n=12):
    """Lines from the nozzle to the footprint ring, while the spray is on."""
    import mujoco
    scn = viewer.user_scn
    scn.ngeom = 0
    if cell.t == 0 or cell.schedule.spray[cell.t - 1] <= 0:
        return
    p, a = cell.nozzle_pose()
    dist = cell.gun.optimal_distance
    r = dist * np.tan(np.radians(cell.gun.cone_half_angle_deg))
    e1 = np.cross(a, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    for k in range(n):
        th = 2 * np.pi * k / n
        end = p + a * dist + r * (np.cos(th) * e1 + np.sin(th) * e2)
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, p, end)
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", choices=["flat", "door", "curved"], default="flat")
    ap.add_argument("--region", default="full", help='e.g. "upper right"')
    ap.add_argument("--mode", choices=["dynamic", "kinematic"], default="dynamic")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--headless", action="store_true", help="no window; print metrics")
    args = ap.parse_args()

    panel = make_panel(args.panel)
    cell = PaintCell(panel, panel.region_mask(args.region), mode=args.mode)
    print(f"{args.panel} panel, region '{args.region}', {args.mode} mode: "
          f"{len(cell)} ticks ({len(cell) / CONTROL_HZ:.1f} s)")

    if args.headless:
        t0 = time.perf_counter()
        cell.run()
        print_summary(cell, time.perf_counter() - t0)
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(cell.model, cell.data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.55]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, -145, -18
        viewer.update_texture(cell.tex_id)
        t0 = time.perf_counter()
        period = 1.0 / (CONTROL_HZ * args.speed)
        next_t = t0
        while viewer.is_running() and not cell.done:
            with viewer.lock():
                cell.step()
                cell.update_texture()
                spray_cone(viewer, cell)
            viewer.update_texture(cell.tex_id)
            viewer.sync()
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))
        print_summary(cell, time.perf_counter() - t0)
        if viewer.is_running():
            print("\nDone. Close the viewer window to exit.")
            with viewer.lock():
                viewer.user_scn.ngeom = 0
            viewer.sync()
            while viewer.is_running():
                time.sleep(0.05)


if __name__ == "__main__":
    main()
