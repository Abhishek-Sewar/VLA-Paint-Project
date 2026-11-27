"""Plan -> simulate -> score -> validate for three panel types, and save heatmaps.

    python scripts/demo.py            # writes outputs/coverage_demo.png
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paintbot import (Panel, RasterPlanner, SprayGun, ValidationConfig,  # noqa: E402
                      coverage_report, validate_episode)
from paintbot.planner import simulate  # noqa: E402

TARGET_UM = 50.0


def scenario(name, panel, region):
    gun = SprayGun()
    target = panel.region_mask(region)
    traj = RasterPlanner(gun, target_thickness=TARGET_UM).plan(panel, target)

    t0 = time.perf_counter()
    thickness, dep, emit = simulate(panel, gun, traj)
    sim_s = time.perf_counter() - t0

    rep = coverage_report(panel, thickness, target, TARGET_UM, deposited=dep, emitted=emit)

    # aim point on a paintable cell, for the spray-miss check
    ix, iy = panel.uv_to_index(traj.uv[:, 0], traj.uv[:, 1])
    ix = np.clip(np.round(ix).astype(int), 0, panel.nx - 1)
    iy = np.clip(np.round(iy).astype(int), 0, panel.ny - 1)
    on_panel = panel.contains_uv(traj.uv[:, 0], traj.uv[:, 1]) & panel.paintable[iy, ix]
    val = validate_episode({"spray": traj.spray, "aim_on_panel": on_panel,
                            "coverage": rep["coverage"]}, ValidationConfig())

    print(f"\n== {name}  (region: {region})")
    print(f"   steps={len(traj)}  painting time={traj.duration:.1f}s  "
          f"sim={sim_s * 1000:.0f} ms  ({len(traj) / sim_s:,.0f} steps/s, "
          f"{traj.duration / sim_s:,.0f}x real time)")
    for k, v in rep.items():
        print(f"   {k:22s} {v:.3f}" if isinstance(v, float) else f"   {k:22s} {v}")
    print(f"   validation: {'PASS' if val.passed else 'FAIL ' + str(val.reasons)} {val.stats}")
    return panel, thickness, target, traj


def main():
    flat = Panel()
    door = Panel()
    door.add_cutout(-0.35, 0.08, 0.35, 0.26)
    curved = Panel(curvature=1.5)

    runs = [
        scenario("Flat panel", flat, "full"),
        scenario("Door with window", door, "full"),
        scenario("Flat panel, partial", Panel(), "upper right"),
        scenario("Curved door (R=0.67 m)", curved, "full"),
    ]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping plot)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    titles = ["Flat panel", "Door with window", "Flat panel, upper right", "Curved door"]
    for ax, title, (panel, th, target, traj) in zip(axes.flat, titles, runs):
        ext = [-panel.width / 2, panel.width / 2, -panel.height / 2, panel.height / 2]
        img = np.ma.masked_where(~panel.paintable, th / TARGET_UM)
        im = ax.imshow(img, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1.5)
        ax.contour(panel.U, panel.V, target.astype(float), levels=[0.5],
                   colors="white", linewidths=1.0, linestyles="--")
        on = traj.spray > 0
        ax.plot(traj.uv[on, 0], traj.uv[on, 1], ",", color="red", alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("u (m)")
        ax.set_ylabel("v (m)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="film thickness / target")
    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "coverage_demo.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
