"""Run with `pytest -q` or plain `python tests/test_core.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paintbot import (Panel, RasterPlanner, SprayGun, ValidationConfig,  # noqa: E402
                      coverage_report, validate_episode)
from paintbot.planner import simulate  # noqa: E402


def test_mass_conservation_static_gun():
    """Gun at optimal distance, normal incidence, mid-panel: all paint lands."""
    panel, gun = Panel(1.0, 1.0, 0.002), SprayGun()
    th = np.zeros((panel.ny, panel.nx))
    dep, emit = gun.deposit(panel, th, [0, 0, 0.25], [0, 0, -1], 1.0, 1.0)
    assert abs(dep / emit - 1.0) < 0.01, dep / emit


def test_too_far_wastes_paint():
    panel, gun = Panel(1.5, 1.5, 0.004), SprayGun()
    th = np.zeros((panel.ny, panel.nx))
    dep, emit = gun.deposit(panel, th, [0, 0, 0.45], [0, 0, -1], 1.0, 1.0)
    assert dep / emit < 0.1


def test_window_matches_full_grid():
    for curvature in (0.0, 1.5):
        panel, gun = Panel(1.0, 0.6, 0.005, curvature=curvature), SprayGun()
        planner = RasterPlanner(gun)
        traj = planner.plan(panel, panel.region_mask("full"))
        idx = np.linspace(0, len(traj) - 1, 40).astype(int)
        a = np.zeros((panel.ny, panel.nx))
        b = np.zeros_like(a)
        for i in idx:
            gun.deposit(panel, a, traj.pos[i], traj.dir[i], 1.0, 1 / 30, window=True)
            gun.deposit(panel, b, traj.pos[i], traj.dir[i], 1.0, 1 / 30, window=False)
        rel = np.abs(a - b).sum() / b.sum()
        assert rel < 0.005, (curvature, rel)


def test_flat_full_panel_quality():
    panel, gun = Panel(), SprayGun()
    planner = RasterPlanner(gun, target_thickness=50.0)
    target = panel.region_mask("full")
    th, dep, emit = simulate(panel, gun, planner.plan(panel, target))
    rep = coverage_report(panel, th, target, 50.0, deposited=dep, emitted=emit)
    assert rep["coverage"] > 0.95, rep
    assert rep["pooling"] < 0.01, rep
    assert 0.9 < rep["mean_rel_thickness"] < 1.1, rep


def _door_with_window(margin):
    panel, gun = Panel(), SprayGun()
    panel.add_cutout(-0.35, 0.08, 0.35, 0.26)
    target = panel.region_mask("full")
    planner = RasterPlanner(gun, cutout_margin_sigma=margin)
    th, _, _ = simulate(panel, gun, planner.plan(panel, target))
    return coverage_report(panel, th, target, 50.0)


def test_door_window_tradeoff():
    """Window edges are a real coverage-vs-overspray trade-off with a raster gun."""
    default = _door_with_window(0.5)
    assert default["coverage"] > 0.85, default
    assert default["glass_contamination"] < 0.35, default
    tight = _door_with_window(1.5)
    assert tight["glass_contamination"] < default["glass_contamination"]
    assert tight["coverage"] < default["coverage"]


def test_curved_panel_quality():
    panel, gun = Panel(curvature=1.5), SprayGun()
    target = panel.region_mask("full")
    th, _, _ = simulate(panel, gun, RasterPlanner(gun).plan(panel, target))
    rep = coverage_report(panel, th, target, 50.0)
    assert rep["coverage"] > 0.90, rep


def test_validation_catches_problems():
    T, J = 100, 7
    cfg = ValidationConfig(joint_limits=np.tile([-2.9, 2.9], (J, 1)))
    clean = {
        "collision": np.zeros(T, bool),
        "joint_pos": np.zeros((T, J)),
        "rgb": np.full((T, 8, 8, 3), 120, np.uint8),
        "depth": np.ones((T, 8, 8)),
        "spray": np.ones(T),
        "aim_on_panel": np.ones(T, bool),
        "coverage": 0.93,
    }
    assert validate_episode(clean, cfg).passed

    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in clean.items()}
    bad["joint_pos"][50, 3] = 3.5
    bad["depth"][10, 2, 2] = np.nan
    bad["rgb"][70] = 0
    bad["aim_on_panel"][:40] = False
    res = validate_episode(bad, cfg)
    assert set(res.reasons) == {"joint_limit", "depth_nan", "black_frame", "spray_miss"}, res


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
