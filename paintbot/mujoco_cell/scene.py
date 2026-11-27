"""Build the MuJoCo painting cell: Franka Panda + spray gun in front of a panel.

World frame: z up, Panda base at the origin, robot faces +x.
The panel stands vertically in front of the robot like a car door. Panel
coordinates (see panel.py) map to world as

    u (across)      -> -y   (so +u is on the robot's right, as panel.py describes)
    v (up)          -> +z
    panel normal +z -> -x   (faces the robot)

`PanelPose` does that mapping, so the planner, BatchedPaintSim and metrics keep
working in panel coordinates and only this package knows about the world.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..panel import Panel
from . import assets

# columns are the panel u, v, n axes expressed in world coordinates
_R_WORLD_PANEL = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])

CONTROL_HZ = 30.0
SUBSTEPS = 20                      # physics steps per control tick
GUN_HEIGHT = 0.06                  # barrel axis, out from the flange (flange +z)
GUN_REACH = 0.15                   # nozzle tip, sideways from the flange axis (flange -y)

PANEL_TEX = "paint_tex"
PANEL_GEOM = "panel_surface"
PANEL_COLLISION = "panel_collision"
NOZZLE_SITE = "nozzle"
# gun geoms use collision bit 2: they touch the panel (bits 1|2) but not the arm (bit 1)
_GUN_COL = dict(contype=2, conaffinity=2)


@dataclass
class PanelPose:
    """Rigid transform panel frame -> world frame (panel centre at `center`)."""
    center: np.ndarray = field(default_factory=lambda: np.array([0.70, 0.0, 0.55]))
    R: np.ndarray = field(default_factory=lambda: _R_WORLD_PANEL.copy())

    def to_world(self, p):
        return np.asarray(p, dtype=float) @ self.R.T + self.center

    def dir_to_world(self, d):
        return np.asarray(d, dtype=float) @ self.R.T

    def to_panel(self, p):
        return (np.asarray(p, dtype=float) - self.center) @ self.R

    def dir_to_panel(self, d):
        return np.asarray(d, dtype=float) @ self.R

    @property
    def quat(self) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.R.reshape(-1))
        return q


def _panel_uv_grid(panel: Panel, step: int = 4):
    """(u, v) mesh-vertex grid, one vertex every `step` cells (2 cm by default)."""
    nx = max(2, int(np.ceil(panel.nx / step)) + 1)
    ny = max(2, int(np.ceil(panel.ny / step)) + 1)
    return np.meshgrid(np.linspace(-panel.width / 2, panel.width / 2, nx),
                       np.linspace(-panel.height / 2, panel.height / 2, ny))


def _panel_mesh(panel: Panel, step: int = 4):
    """Front-surface triangle mesh (panel frame) with UVs.

    UV convention: tex u = (u + W/2) / W, tex v = (v + H/2) / H. texture.py
    writes texels to match, and tests/test_mujoco_cell.py checks it on screen.
    """
    U, V = _panel_uv_grid(panel, step)
    ny, nx = U.shape
    P, _ = panel.surface(U, V)
    verts = P.reshape(-1, 3)
    tc = np.stack([(U + panel.width / 2) / panel.width,
                   (V + panel.height / 2) / panel.height], axis=-1).reshape(-1, 2)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    # counter-clockwise seen from +z (the painted side)
    faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return verts, tc, faces


def _panel_slab(panel: Panel, thickness: float = 0.01):
    """Closed vertex cloud (front + back) for a convex collision hull."""
    us = np.linspace(-panel.width / 2, panel.width / 2, 21)
    vs = np.array([-panel.height / 2, panel.height / 2])
    U, V = np.meshgrid(us, vs)
    P, N = panel.surface(U, V)
    front = P.reshape(-1, 3)
    back = (P - N * thickness).reshape(-1, 3)
    return np.concatenate([front, back])


RAIL_RANGE = 0.8          # carriage travel, +- metres along y
PEDESTAL = 0.3            # rail top height. The gun hangs ~0.17 m below the wrist,
                          # so from floor level the curved door's top corners were
                          # at the limit of the Panda's reach


def _with_rail(xml: str) -> str:
    """Put the Panda on a carriage that slides along world y (a 7th-axis rail)."""
    head = '<body name="link0" childclass="panda">'
    assert head in xml and xml.count("</worldbody>") == 1
    xml = xml.replace(head, (
        f'<body name="rail_carriage" pos="0 0 {PEDESTAL}">'
        f'<joint name="rail" type="slide" axis="0 1 0" range="-{RAIL_RANGE} {RAIL_RANGE}"'
        ' damping="50" armature="1"/>'
        '<geom type="box" size="0.12 0.12 0.02" pos="0 0 -0.02" rgba="0.2 0.2 0.22 1"'
        ' mass="5" contype="0" conaffinity="0"/>' + head))
    xml = xml.replace("</worldbody>", "</body></worldbody>")
    xml = xml.replace("<actuator>", '<actuator><position name="rail_act" joint="rail"'
                      f' kp="20000" kv="1500" ctrlrange="-{RAIL_RANGE} {RAIL_RANGE}"/>')
    xml = xml.replace('qpos="0 0 0', 'qpos="0 0 0 0').replace('ctrl="0 0 0', 'ctrl="0 0 0 0')
    return xml


def build_scene(panel: Panel, pose: PanelPose | None = None,
                rail: bool = True):
    """Return (model, pose). The panel texture starts as bare metal.

    rail=True (default) mounts the Panda on a +-0.8 m linear rail along y. Without
    it a fixed Panda reaches only the middle 60% of the 1.0 m curved door: near the
    edges the surface normal turns ~43 deg and pushes the wrist ~0.8 m sideways.
    """
    pose = pose or PanelPose()
    xml_path = assets.panda_xml()
    if rail:
        spec = mujoco.MjSpec.from_string(_with_rail(xml_path.read_text()))
        spec.meshdir = str(xml_path.parent / "assets")
    else:
        spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.modelname = "paintbot_cell"
    spec.option.timestep = 1.0 / (CONTROL_HZ * SUBSTEPS)

    # gravity compensation: the Panda's PD gains then track instead of sagging
    for b in spec.bodies:
        if b.name.startswith("link"):
            b.gravcomp = 1.0

    # ---------------------------------------------------------------- spray gun
    # Flange coordinates: +z points out of the flange. The barrel points along
    # flange -y, at 90 deg to the flange like a real spray gun, so the wrist can
    # stay near its flange-down `home` posture while spraying horizontally (a gun
    # in line with the flange drove joint 6 into its limit).
    flange = spec.body("attachment")
    gun = flange.add_body(name="spray_gun", gravcomp=1.0)
    grey = [0.25, 0.25, 0.28, 1]
    rx90 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]     # local z -> flange -y
    h = GUN_HEIGHT
    gun.add_geom(name="gun_mount", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[0.03, 0.01, 0], pos=[0, 0, 0.01], rgba=grey, mass=0.15, **_GUN_COL)
    gun.add_geom(name="gun_body", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[0.02, 0.035, 0.03], pos=[0, 0, h - 0.01],
                 rgba=[0.15, 0.35, 0.6, 1], mass=0.3, **_GUN_COL)
    gun.add_geom(name="gun_cup", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[0.022, 0.025, 0], pos=[0, 0.01, h + 0.045],
                 rgba=[0.8, 0.8, 0.82, 1], mass=0.2, **_GUN_COL)
    gun.add_geom(name="gun_barrel", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[0, -0.03, h, 0, -(GUN_REACH - 0.012), h], size=[0.009, 0, 0],
                 rgba=grey, mass=0.05, **_GUN_COL)
    gun.add_geom(name="gun_tip", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[0.005, 0.006, 0], pos=[0, -(GUN_REACH - 0.006), h], quat=rx90,
                 rgba=[0.7, 0.55, 0.2, 1], mass=0.01, **_GUN_COL)
    gun.add_site(name=NOZZLE_SITE, pos=[0, -GUN_REACH, h], quat=rx90, size=[0.004, 0, 0],
                 rgba=[1, 1, 0, 1])

    # -------------------------------------------------------------------- panel
    verts, tc, faces = _panel_mesh(panel)
    spec.add_mesh(name="panel_mesh", uservert=verts.ravel().tolist(),
                  usertexcoord=tc.ravel().tolist(), userface=faces.ravel().tolist(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    spec.add_mesh(name="panel_slab", uservert=_panel_slab(panel).ravel().tolist())
    # back face (reversed winding, 3 mm behind) so the panel doesn't vanish when
    # the viewer orbits behind it; the textured mesh is one-sided
    _, N_v = panel.surface(*_panel_uv_grid(panel))
    back = verts - N_v.reshape(-1, 3) * 0.003
    spec.add_mesh(name="panel_back_mesh", uservert=back.ravel().tolist(),
                  userface=faces[:, ::-1].ravel().tolist(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)

    from .texture import bare_texture   # local import keeps texture.py mujoco-free
    tex = bare_texture(panel)
    spec.add_texture(name=PANEL_TEX, type=mujoco.mjtTexture.mjTEXTURE_2D,
                     width=tex.shape[1], height=tex.shape[0], nchannel=3,
                     data=tex.ravel().tolist())
    mat = spec.add_material(name="panel_mat", specular=0.3, shininess=0.4)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = PANEL_TEX

    stand = spec.worldbody.add_body(name="panel_stand", pos=pose.center.tolist(),
                                    quat=pose.quat.tolist())
    stand.add_geom(name=PANEL_GEOM, type=mujoco.mjtGeom.mjGEOM_MESH, meshname="panel_mesh",
                   material="panel_mat", contype=0, conaffinity=0, group=1)
    stand.add_geom(name="panel_back", type=mujoco.mjtGeom.mjGEOM_MESH,
                   meshname="panel_back_mesh", rgba=[0.45, 0.46, 0.48, 1],
                   contype=0, conaffinity=0, group=1)
    stand.add_geom(name=PANEL_COLLISION, type=mujoco.mjtGeom.mjGEOM_MESH,
                   meshname="panel_slab", contype=3, conaffinity=3, group=3,
                   rgba=[0, 0, 0, 0])
    # cutouts drawn as glass just behind the surface (thin boxes, visual only)
    for i, (u0, v0, u1, v1) in enumerate(panel.cutouts):
        uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
        P, N = panel.surface(np.array(uc), np.array(vc))
        n = np.asarray(N)
        yaw = np.arctan2(n[0], n[2])             # rotate box about panel v axis
        q = np.array([np.cos(yaw / 2), 0, np.sin(yaw / 2), 0])
        stand.add_geom(name=f"glass_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[(u1 - u0) / 2, (v1 - v0) / 2, 0.002],
                       pos=(np.asarray(P) - n * 0.004).tolist(), quat=q.tolist(),
                       rgba=[0.45, 0.65, 0.9, 0.35], contype=0, conaffinity=0, group=1)
    # two posts from the floor to the panel's bottom corners, just behind the surface
    for side, name in ((-1, "post_l"), (1, "post_r")):
        P, N = panel.surface(np.array(side * 0.45 * panel.width), np.array(-panel.height / 2))
        foot = pose.to_world(np.asarray(P) - np.asarray(N) * 0.03)
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.015, foot[2] / 2],
            pos=[foot[0], foot[1], foot[2] / 2], rgba=[0.3, 0.3, 0.3, 1],
            contype=0, conaffinity=0, group=1)

    if rail:   # the rail beam the carriage runs on
        spec.worldbody.add_geom(
            name="rail_beam", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.10, RAIL_RANGE + 0.15, (PEDESTAL - 0.04) / 2],
            pos=[0, 0, (PEDESTAL - 0.04) / 2], rgba=[0.35, 0.36, 0.38, 1],
            contype=0, conaffinity=0, group=1)

    # ------------------------------------------------------------- environment
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=256, height=256,
                     rgb1=[0.42, 0.44, 0.46], rgb2=[0.36, 0.38, 0.40])
    fmat = spec.add_material(name="floor_mat", texrepeat=[6, 6], reflectance=0.05)
    fmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[3, 3, 0.05], material="floor_mat", contype=0,
                            conaffinity=0)
    spec.worldbody.add_light(name="key", pos=[0.3, 1.0, 2.2], dir=[0.2, -0.5, -1.0],
                             diffuse=[0.7, 0.7, 0.7])
    spec.worldbody.add_light(name="fill", pos=[-0.8, -1.0, 2.0], dir=[0.5, 0.5, -1.0],
                             diffuse=[0.35, 0.35, 0.35])
    spec.worldbody.add_camera(name="overview", pos=[-0.9, -1.1, 1.25],
                              xyaxes=[0.77, -0.64, 0, 0.25, 0.30, 0.92])
    spec.worldbody.add_camera(name="panel_view", pos=[-0.55, 0.0, 0.55],
                              xyaxes=[0, -1, 0, 0.1, 0, 1])
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    model = spec.compile()
    return model, pose
