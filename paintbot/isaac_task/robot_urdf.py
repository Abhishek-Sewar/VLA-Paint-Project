"""One URDF for the whole painting robot: rail + FANUC arm + spray gun + nozzle frame.

Isaac Sim imports this (UrdfConverter). It reproduces the MuJoCo cell exactly
(same rail, pedestal, gun geometry and nozzle frame as mujoco_cell/scene.py and
robots.py), and tests/test_isaac_task.py checks that by loading this URDF in
MuJoCo and comparing nozzle poses with the verified cell.

    rail_base (fixed to world) -[rail: prismatic, y]-> carriage -> base_link -> J1..J6
    -> flange -> fanuc_flange -> spray_gun -> nozzle   (nozzle +z = spray direction)

No Isaac or MuJoCo imports, so it runs anywhere (laptop or cloud box).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from ..cell_layout import GUN_HEIGHT, GUN_REACH, LAYOUTS
from ..mujoco_cell.assets import fanuc_urdf  # stdlib + xacro only, no mujoco

_SKIP_LINKS = {"world", "tool0"}
_SKIP_JOINTS = {"base_joint", "child_joint"}


def _origin(parent, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    ET.SubElement(parent, "origin", xyz=" ".join(f"{v:.12g}" for v in xyz),
                  rpy=" ".join(f"{v:.12g}" for v in rpy))


def _inertial(link, mass, xyz=(0, 0, 0), i=1e-4):
    ine = ET.SubElement(link, "inertial")
    _origin(ine, xyz)
    ET.SubElement(ine, "mass", value=f"{mass:g}")
    ET.SubElement(ine, "inertia", ixx=f"{i:g}", iyy=f"{i:g}", izz=f"{i:g}",
                  ixy="0", ixz="0", iyz="0")


def _shape(link, kind, geom: dict, xyz=(0, 0, 0), rpy=(0, 0, 0), rgba=None, name=None):
    """Add matching <visual> and <collision> elements."""
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag, **({"name": f"{name}_{tag}"} if name else {}))
        _origin(el, xyz, rpy)
        g = ET.SubElement(ET.SubElement(el, "geometry"), kind,
                          **{k: " ".join(f"{x:.6g}" for x in v) if isinstance(v, (tuple, list))
                             else f"{v:.6g}" for k, v in geom.items()})
        if tag == "visual" and rgba is not None:
            mat = ET.SubElement(el, "material", name=f"{name or kind}_mat")
            ET.SubElement(mat, "color", rgba=" ".join(f"{c:g}" for c in rgba))
        del g


def _link(robot, name):
    return ET.SubElement(robot, "link", name=name)


def _joint(robot, name, kind, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0), **extra):
    j = ET.SubElement(robot, "joint", name=name, type=kind)
    _origin(j, xyz, rpy)
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    for tag, attrs in extra.items():
        ET.SubElement(j, tag, **attrs)
    return j


def build_cell_urdf(robot: str = "crx10ia_l", out: Path | None = None) -> Path:
    """Write <robot>_paintcell.urdf next to the FANUC build output and return its path."""
    lay = LAYOUTS[robot]
    src = ET.parse(fanuc_urdf(robot)).getroot()
    cell = ET.Element("robot", name=f"{robot}_paintcell")

    # --- rail -----------------------------------------------------------------
    base = _link(cell, "rail_base")
    _inertial(base, 50.0, i=1.0)
    if lay.rail:
        _shape(base, "box", {"size": (0.20, 2 * lay.rail_range + 0.3, max(lay.pedestal - 0.04, 0.01))},
               xyz=(0, 0, max(lay.pedestal - 0.04, 0.01) / 2), rgba=(0.35, 0.36, 0.38, 1),
               name="rail_beam")
        carriage = _link(cell, "carriage")
        _inertial(carriage, 5.0, i=0.05)
        _shape(carriage, "box", {"size": (0.24, 0.24, 0.04)}, xyz=(0, 0, -0.02),
               rgba=(0.2, 0.2, 0.22, 1), name="carriage_plate")
        r = lay.rail_range
        _joint(cell, "rail", "prismatic", "rail_base", "carriage", xyz=(0, 0, lay.pedestal),
               axis={"xyz": "0 1 0"},
               limit={"lower": f"{-r:g}", "upper": f"{r:g}", "effort": "5000", "velocity": "1.0"})
        _joint(cell, "carriage_to_base", "fixed", "carriage", "base_link")
    else:
        _joint(cell, "mount_to_base", "fixed", "rail_base", "base_link", xyz=(0, 0, lay.pedestal))

    # --- FANUC arm (copied as-is, minus its world/tool0 wrapper) ----------------
    for el in src:
        if el.tag == "link" and el.get("name") in _SKIP_LINKS:
            continue
        if el.tag == "joint" and el.get("name") in _SKIP_JOINTS:
            continue
        cell.append(el)

    # --- spray gun: same geometry as mujoco_cell/scene.py -----------------------
    gun = _link(cell, "spray_gun")
    _inertial(gun, 0.71, xyz=(0, -0.02, GUN_HEIGHT), i=2e-3)
    h, reach = GUN_HEIGHT, GUN_REACH
    rx90 = (math.pi / 2, 0, 0)                     # local z -> flange -y
    grey, blue, alu = (0.25, 0.25, 0.28, 1), (0.15, 0.35, 0.6, 1), (0.8, 0.8, 0.82, 1)
    _shape(gun, "cylinder", {"radius": 0.03, "length": 0.02}, xyz=(0, 0, 0.01), rgba=grey,
           name="gun_mount")
    _shape(gun, "box", {"size": (0.04, 0.07, 0.06)}, xyz=(0, 0, h - 0.01), rgba=blue,
           name="gun_body")
    _shape(gun, "cylinder", {"radius": 0.022, "length": 0.05}, xyz=(0, 0.01, h + 0.045),
           rgba=alu, name="gun_cup")
    barrel_len = (reach - 0.012) - 0.03
    _shape(gun, "cylinder", {"radius": 0.009, "length": barrel_len},
           xyz=(0, -(0.03 + barrel_len / 2), h), rpy=rx90, rgba=grey, name="gun_barrel")
    _shape(gun, "cylinder", {"radius": 0.005, "length": 0.012}, xyz=(0, -(reach - 0.006), h),
           rpy=rx90, rgba=(0.7, 0.55, 0.2, 1), name="gun_tip")
    _joint(cell, "flange_to_gun", "fixed", lay.flange_link, "spray_gun")

    nozzle = _link(cell, "nozzle")
    _inertial(nozzle, 0.001, i=1e-7)
    _joint(cell, "gun_to_nozzle", "fixed", "spray_gun", "nozzle", xyz=(0, -reach, h), rpy=rx90)

    out = Path(out or fanuc_urdf(robot).with_name(f"{robot}_paintcell.urdf"))
    ET.indent(cell)
    out.write_text('<?xml version="1.0"?>\n' + ET.tostring(cell, encoding="unicode"))
    return out
