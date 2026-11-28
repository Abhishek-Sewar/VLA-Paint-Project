"""Checks the MuJoCo painting cell (robot arm on a rail + spray gun + panel).

    python tests/test_mujoco_cell.py      # or: python -m pytest -q
Runs every robot check on the Franka Panda and, if fetched, the FANUC CRX-10iA/L
(python scripts/fetch_menagerie.py --all). Skips if mujoco or the Panda is missing.
"""
import functools
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg, allow_module_level=True)
    except ImportError:
        print(f"{msg}; skipping")
        sys.exit(0)


try:
    import mujoco
    import torch
except ImportError:  # pragma: no cover
    _skip("mujoco or torch not installed")

from paintbot.mujoco_cell import assets  # noqa: E402

if not assets.available():  # pragma: no cover
    _skip("Menagerie Panda not fetched (python scripts/fetch_menagerie.py)")

from paintbot import Panel  # noqa: E402
from paintbot.mujoco_cell import PaintCell, build_scene  # noqa: E402
from paintbot.mujoco_cell.ik import NozzleIK  # noqa: E402
from paintbot.mujoco_cell.scene import NOZZLE_SITE, PANEL_GEOM  # noqa: E402
from paintbot.mujoco_cell.texture import BARE, GLASS, thickness_rgb, write_texture  # noqa: E402
from paintbot.torch_sim import BatchedPaintSim  # noqa: E402

DOOR_WINDOW = (-0.35, 0.08, 0.35, 0.26)
ROBOTS = ["panda"] + (["crx10ia_l"] if assets.available("crx10ia_l") else [])
if len(ROBOTS) == 1:  # pragma: no cover
    print("   (FANUC CRX not fetched; testing the Panda only)")


def _panel(kind):
    if kind == "curved":
        return Panel(curvature=1.5)
    p = Panel()
    if kind == "door":
        p.add_cutout(*DOOR_WINDOW)
    return p


@functools.lru_cache(maxsize=None)
def _finished_cell(kind, mode, robot="panda", region="full"):
    p = _panel(kind)
    return PaintCell(p, p.region_mask(region), mode=mode, dtype=torch.float64,
                     robot=robot).run()


def _ik(m, cfg):
    home = np.asarray(cfg.home, float)
    return NozzleIK(m, home=np.concatenate([[0.0], home]) if cfg.rail else home)


def _direct_sim(cell):
    """BatchedPaintSim fed the planner trajectory directly (no robot)."""
    tr, p = cell.traj, cell.panel
    sim = BatchedPaintSim(1, dtype=torch.float64)
    sim.set_panel([0], curvature=p.curvature, cutouts=p.cutouts)
    sim.set_target([0], cell.target)
    for t in range(len(tr)):
        sim.step(tr.pos[t][None], tr.dir[t][None], tr.spray[t:t + 1])
    return sim


def test_core_does_not_import_mujoco():
    code = ("import sys; sys.path.insert(0, r'%s'); import paintbot, paintbot.torch_sim; "
            "assert 'mujoco' not in sys.modules" % os.path.abspath(ROOT))
    subprocess.run([sys.executable, "-c", code], check=True)


def test_panel_mesh_matches_panel_geometry():
    for kind in ("flat", "door", "curved"):
        p = _panel(kind)
        m, pose, _ = build_scene(p)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        g = m.geom(PANEL_GEOM).id
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        world = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        local = pose.to_panel(world)                       # back to panel coordinates
        if p.curvature == 0:
            assert np.abs(local[:, 2]).max() < 1e-6
            assert np.allclose(np.abs(local[:, :2]).max(0), [p.width / 2, p.height / 2])
        else:
            R = 1 / p.curvature
            r = np.hypot(local[:, 0], local[:, 2] + R)
            assert np.abs(r - R).max() < 1e-6, kind
        # the corner at panel (+u, +v) is on the robot's upper right: -y, +z in world
        P, _ = p.surface(np.array(p.width / 2), np.array(p.height / 2))
        corner = pose.to_world(np.asarray(P))
        assert np.linalg.norm(world - corner, axis=1).min() < 1e-6
        assert corner[1] < 0 and corner[2] > pose.center[2]


def test_panel_faces_robot_and_nozzle_axis_is_spray_direction():
    for robot in ROBOTS:
        p = Panel()
        m, pose, _ = build_scene(p, robot=robot)
        assert np.allclose(pose.dir_to_world([0, 0, 1]), [-1, 0, 0])   # normal faces robot
        assert np.allclose(pose.dir_to_world([1, 0, 0]), [0, -1, 0])   # +u = robot's right
        assert np.allclose(pose.to_panel(pose.to_world([0.1, -0.2, 0.3])), [0.1, -0.2, 0.3])
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        s = m.site(NOZZLE_SITE).id
        z = d.site_xmat[s].reshape(3, 3)[:, 2]
        tip = d.geom_xpos[m.geom("gun_tip").id]
        barrel = d.geom_xpos[m.geom("gun_barrel").id]
        nozzle = d.site_xpos[s]
        # barrel -> tip -> nozzle all lie along the nozzle's +z axis, in that order
        for a, b in ((barrel, tip), (tip, nozzle)):
            u = (b - a) / np.linalg.norm(b - a)
            assert u @ z > 0.999, robot


def test_ik_reaches_centre_and_corners():
    for robot in ROBOTS:
        for kind in ("flat", "curved"):
            p = _panel(kind)
            m, pose, cfg = build_scene(p, robot=robot)
            ik = _ik(m, cfg)
            for u, v in [(0, 0), (-.5, .3), (.5, .3), (-.5, -.3), (.5, -.3)]:
                P, N = p.surface(np.array(u), np.array(v))
                tgt = pose.to_world(np.asarray(P) + 0.25 * np.asarray(N))
                q, pe, ae = ik.solve(tgt, pose.dir_to_world(-np.asarray(N)))
                assert pe < 1e-3 and ae < np.radians(0.5), (robot, kind, u, v, pe, ae)
                assert np.all(q >= m.jnt_range[:, 0] - 1e-9) and np.all(q <= m.jnt_range[:, 1] + 1e-9)


def test_full_trajectories_reachable_and_continuous():
    # PaintCell raises if IK would jump; also check the solved path is exact
    for robot in ROBOTS:
        for kind, region in (("flat", "full"), ("door", "full"), ("curved", "full"),
                             ("flat", "upper right")):
            p = _panel(kind)
            cell = PaintCell(p, p.region_mask(region), mode="kinematic", robot=robot)
            S = cell.schedule
            assert S.ik_pos_err.max() < 1e-4 and S.ik_ang_err.max() < 1e-3, (robot, kind, region)
            assert np.abs(np.diff(S.q, axis=0)).max() < 0.1, (robot, kind, region)


def test_kinematic_matches_direct_sim():
    for robot, kind in [(r, k) for r in ROBOTS for k in ("flat", "curved")]:
        cell = _finished_cell(kind, "kinematic", robot)
        ref = _direct_sim(cell)
        th, rth = cell.thickness, ref.thickness[0].numpy()
        assert np.abs(th - rth).max() / rth.max() < 1e-3, (robot, kind)
        m, r = ref.metrics(50.0), cell.report()
        for key in ("coverage", "in_spec", "uniformity_cv", "transfer_efficiency"):
            assert abs(m[key][0].item() - r[key]) < 1e-3, (robot, kind, key)


def test_dynamic_tracks_paints_and_validates():
    for robot, kind in [(r, k) for r in ROBOTS for k in ("flat", "door", "curved")]:
        cell = _finished_cell(kind, "dynamic", robot)
        ts = cell.tracking_stats()
        # measured: Panda max 4.7-6.0 mm / mean 0.9-1.4 mm / 0.2-0.5 deg,
        #           CRX   max 5.5-8.6 mm / mean 0.9-1.6 mm / 0.1-0.5 deg
        assert ts["max_pos_err_mm"] < 10.0 and ts["mean_pos_err_mm"] < 2.5, (robot, kind, ts)
        assert ts["max_ang_err_deg"] < 1.0, (robot, kind, ts)
        ref_cov = _direct_sim(cell).metrics(50.0)["coverage"][0].item()
        assert abs(cell.report()["coverage"] - ref_cov) < 0.02, (robot, kind)
        assert not cell.log["collision"].any(), (robot, kind)
        res = cell.validate()
        assert res.passed, (robot, kind, res.reasons, res.stats)


def test_texture_colours_and_layout():
    th = np.zeros((4, 6))
    paintable = np.ones((4, 6), dtype=bool)
    paintable[0, 0] = False
    th[3, 5] = 50.0                                  # top-right cell (row 3 = top)
    th[1, 2] = 100.0                                 # pooled
    rgb = thickness_rgb(th, paintable, 50.0)
    assert np.allclose(rgb[2, 2] / 255, BARE, atol=0.01)
    assert np.allclose(rgb[0, 0] / 255, GLASS, atol=0.01)
    assert rgb[3, 5, 0] > 150 and rgb[3, 5, 1] < 80  # painted red
    assert rgb[1, 2].sum() < rgb[3, 5].sum()         # pooling drawn darker

    class FakeModel:                                 # write_texture row order
        tex_adr = np.array([3])
        tex_height, tex_width, tex_nchannel = np.array([4]), np.array([6]), np.array([3])
        tex_data = np.zeros(3 + 4 * 6 * 3, dtype=np.uint8)
    fm = FakeModel()
    write_texture(fm, 0, rgb)
    assert np.array_equal(fm.tex_data[3:].reshape(4, 6, 3), rgb)


def test_texture_renders_on_robots_upper_right():
    p = Panel()
    cell = PaintCell(p, mode="kinematic")
    U, V = np.meshgrid(p.u, p.v)
    blob = (U - 0.3) ** 2 + (V - 0.15) ** 2 < 0.06 ** 2      # u right, v up
    cell.sim.thickness[0][torch.as_tensor(blob)] = 60.0
    cell.update_texture()
    try:
        r = mujoco.Renderer(cell.model, 360, 640)
    except Exception as e:  # pragma: no cover - no OpenGL context available
        print(f"   (no OpenGL context: {e}; skipped render check)")
        return
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1                                     # panel only
    r.update_scene(cell.data, camera="panel_view", scene_option=opt)
    img = r.render().astype(int)
    r.close()
    red = (img[..., 0] - img[..., 1]) > 80
    ys, xs = np.nonzero(red)
    assert red.sum() > 200
    # camera looks at the panel from the robot's side: image right = +u, image up = +v
    assert xs.mean() / img.shape[1] > 0.6 and ys.mean() / img.shape[0] < 0.4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
