"""Checks the Isaac Sim task's inputs without Isaac Sim (which can't run on the laptop).

    python tests/test_isaac_task.py      # or: python -m pytest -q

The Isaac env itself (paintbot/isaac_task/env.py) only runs on the cloud GPU. What it
loads is checked here: the robot URDF must match the verified MuJoCo cell, the
expert's joint path must reproduce the planned nozzle path through that URDF, and
the panel USD must match panel.py. Needs mujoco, usd-core and the FANUC model
(python scripts/fetch_menagerie.py --fanuc); skips what is missing.
"""
import functools
import os
import py_compile
import subprocess
import sys
import tempfile

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
except ImportError:  # pragma: no cover
    _skip("mujoco not installed")

from paintbot.cell_layout import make_panel  # noqa: E402
from paintbot.mujoco_cell import assets  # noqa: E402

if not assets.available("crx10ia_l"):  # pragma: no cover
    _skip("FANUC CRX not fetched (python scripts/fetch_menagerie.py --fanuc)")

from paintbot.isaac_task.expert import export_expert, load_expert  # noqa: E402
from paintbot.isaac_task.robot_urdf import build_cell_urdf  # noqa: E402

try:
    from pxr import Usd, UsdGeom  # noqa: F401
    HAVE_PXR = True
except ImportError:  # pragma: no cover
    HAVE_PXR = False

TMP = tempfile.mkdtemp(prefix="paintbot_isaac_")


@functools.lru_cache(maxsize=None)
def _urdf_model():
    """The Isaac robot URDF, loaded in MuJoCo."""
    u = build_cell_urdf("crx10ia_l", out=os.path.join(TMP, "cell.urdf")).read_text()
    tag = ('<mujoco><compiler fusestatic="false" discardvisual="true" strippath="false" '
           'balanceinertia="true"/></mujoco>')
    u = u.replace('<robot name="crx10ia_l_paintcell">', '<robot name="crx10ia_l_paintcell">' + tag, 1)
    m = mujoco.MjSpec.from_string(u).compile()
    return m, mujoco.MjData(m)


@functools.lru_cache(maxsize=None)
def _expert(kind):
    return load_expert(export_expert(kind, "crx10ia_l", out=os.path.join(TMP, f"{kind}.npz")))


def test_isaac_side_imports_without_mujoco():
    code = ("import sys; sys.path.insert(0, r'%s'); "
            "import paintbot.cell_layout, paintbot.paint_colors, paintbot.isaac_task.robot_urdf, "
            "paintbot.isaac_task.expert, paintbot.isaac_task.panel_usd; "
            "assert 'mujoco' not in sys.modules" % os.path.abspath(ROOT))
    subprocess.run([sys.executable, "-c", code], check=True)


def test_isaac_files_compile():
    for f in ("paintbot/isaac_task/env.py", "scripts/isaac/run_paint.py",
              "scripts/isaac/prepare_assets.py"):
        py_compile.compile(os.path.join(ROOT, f), doraise=True)


def test_robot_urdf_matches_mujoco_cell():
    from paintbot.mujoco_cell import PaintCell
    mu, du = _urdf_model()
    assert [mu.joint(i).name for i in range(mu.njnt)] == ["rail", "J1", "J2", "J3", "J4", "J5", "J6"]
    cell = PaintCell(make_panel("flat"), mode="kinematic", robot="crx10ia_l")
    mc, dc = cell.model, cell.data
    s, b = mc.site("nozzle").id, mu.body("nozzle").id
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(mc.jnt_range[:, 0], mc.jnt_range[:, 1])
        dc.qpos[:] = q
        du.qpos[:] = q
        mujoco.mj_kinematics(mc, dc)
        mujoco.mj_kinematics(mu, du)
        assert np.linalg.norm(dc.site_xpos[s] - du.xpos[b]) < 1e-9
        assert np.allclose(dc.site_xmat[s], du.xmat[b], atol=1e-9)
    assert np.allclose(mu.jnt_range, mc.jnt_range)


def test_expert_replays_planned_nozzle_path_through_urdf():
    mu, du = _urdf_model()
    b = mu.body("nozzle").id
    for kind in ("door", "curved"):
        e = _expert(kind)
        on = e["spray"] > 0
        assert on.sum() > 500 and e["meta"]["joint_names"][0] == "rail"
        err_p, err_a = [], []
        for t in np.nonzero(on)[0]:
            du.qpos[:] = e["q"][t]
            mujoco.mj_kinematics(mu, du)
            err_p.append(np.linalg.norm(du.xpos[b] - e["nozzle_pos"][t]))
            err_a.append(1 - du.xmat[b].reshape(3, 3)[:, 2] @ e["nozzle_dir"][t])
        # float32 storage limits this to ~1e-5 m
        assert max(err_p) < 1e-4 and max(err_a) < 1e-6, (kind, max(err_p), max(err_a))
        ref = e["meta"]["mujoco_reference"]
        assert ref["coverage"] > 0.85 and ref["accept"], (kind, ref)


def test_panel_usd_matches_panel_and_takes_colours():
    if not HAVE_PXR:  # pragma: no cover
        print("   (usd-core not installed; skipped USD check)")
        return
    from pxr import Usd, UsdGeom
    from paintbot.isaac_task.panel_usd import set_vertex_colors, vertex_colors, write_panel_usd
    for kind in ("curved", "door"):
        p = make_panel(kind)
        path = write_panel_usd(p, os.path.join(TMP, f"panel_{kind}.usda"))
        stage = Usd.Stage.Open(str(path))
        mesh = stage.GetPrimAtPath("/Panel/surface")
        pts = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
        half_w = p.width / 2
        if p.curvature:
            R = 1 / p.curvature
            assert np.abs(np.hypot(pts[:, 0], pts[:, 2] + R) - R).max() < 1e-5
            half_w = R * np.sin(half_w / R)              # x is the chord, u the arc
        else:
            assert np.abs(pts[:, 2]).max() < 1e-7
        assert np.allclose(pts.min(0)[:2], [-half_w, -p.height / 2], atol=1e-5)
        assert np.allclose(pts.max(0)[:2], [half_w, p.height / 2], atol=1e-5)
        cells = np.array(stage.GetPrimAtPath("/Panel").GetAttribute("paintbot:vertCells").Get()).reshape(-1, 2)
        assert len(cells) == len(pts)
        th = np.zeros((p.ny, p.nx))
        th[p.v > 0.2, :] = 50.0                      # paint the top band
        col = vertex_colors(th, p, cells)
        set_vertex_colors(mesh, col)
        got = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("displayColor").Get())
        assert np.allclose(got, col, atol=1e-6)
        top = pts[:, 1] > 0.25
        assert got[top, 0].mean() > got[~top, 0].mean()  # painted (red) on top only
        if kind == "door":                               # glass drawn blue
            glass = ~p.paintable[cells[:, 0], cells[:, 1]]
            assert glass.any() and (got[glass, 2] > got[glass, 0]).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
