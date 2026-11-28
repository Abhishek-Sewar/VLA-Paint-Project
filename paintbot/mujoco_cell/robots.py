"""Robot configurations for the MuJoCo cell. Add a robot by adding a CellLayout (in
paintbot/cell_layout.py, shared with Isaac Sim) and a RobotConfig here.

Each robot is loaded as its own MjSpec (base at the origin) and attached into the
cell on a pedestal or a linear rail (scene.py). The spray gun is mounted on the
layout's flange link, whose +z axis must point out of the tool flange.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco

from ..cell_layout import LAYOUTS, CellLayout
from . import assets


@dataclass(frozen=True)
class RobotConfig:
    layout: CellLayout
    load: Callable[[], mujoco.MjSpec]     # arm only, base at origin
    # position servo gains for arms whose model has no actuators (None = keep model's)
    kp: tuple | None = None
    kv: tuple | None = None
    armature: float = 0.0
    rgba: tuple | None = None             # recolour collision-mesh visuals

    # layout fields, for convenience
    name = property(lambda self: self.layout.robot)
    flange_body = property(lambda self: self.layout.flange_link)
    home = property(lambda self: self.layout.home)
    rail = property(lambda self: self.layout.rail)
    rail_range = property(lambda self: self.layout.rail_range)
    pedestal = property(lambda self: self.layout.pedestal)
    panel_center = property(lambda self: self.layout.panel_center)


def _load_panda() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(assets.panda_xml()))


def _load_fanuc(model: str) -> Callable[[], mujoco.MjSpec]:
    def load() -> mujoco.MjSpec:
        urdf = assets.fanuc_urdf(model).read_text()
        # keep the flange frames as bodies; visuals are .dae (MuJoCo can't read
        # them), so the collision STLs double as visuals
        tag = ('<mujoco><compiler fusestatic="false" discardvisual="true" '
               'strippath="false" balanceinertia="true"/></mujoco>')
        urdf = urdf.replace(f'<robot name="{model}">', f'<robot name="{model}">{tag}', 1)
        return mujoco.MjSpec.from_string(urdf)
    return load


ROBOTS: dict[str, RobotConfig] = {
    "panda": RobotConfig(LAYOUTS["panda"], _load_panda),
    "crx10ia_l": RobotConfig(
        LAYOUTS["crx10ia_l"], _load_fanuc("crx10ia_l"),
        kp=(8000, 8000, 5000, 1500, 1500, 800), kv=(800, 800, 500, 150, 150, 80),
        armature=0.1, rgba=(0.93, 0.93, 0.91, 1.0)),
}


def get(name: str) -> RobotConfig:
    try:
        return ROBOTS[name]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; choose from {sorted(ROBOTS)}") from None
