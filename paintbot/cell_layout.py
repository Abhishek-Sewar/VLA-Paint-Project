"""Painting-cell layout shared by every simulator front end (MuJoCo, Isaac Sim).

Plain NumPy: where the robot stands, the rail, the panel pose, the spray-gun
geometry. mujoco_cell/ and isaac_task/ both build their scenes from this, so the
cell that is verified locally in MuJoCo is the one Isaac Sim runs.

World frame: z up, robot base (or rail centre) at the origin, robot faces +x. The
panel stands vertically in front of the robot. Panel coordinates (panel.py) map to
world as u -> -y (+u on the robot's right), v -> +z, panel normal -> -x.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# spray gun, in flange coordinates (+z out of the flange). The barrel points along
# flange -y, at 90 deg to the flange like a real spray gun.
GUN_HEIGHT = 0.06          # barrel axis, out from the flange
GUN_REACH = 0.15           # nozzle tip, sideways from the flange axis

# columns are the panel u, v, n axes expressed in world coordinates
_R_WORLD_PANEL = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class CellLayout:
    robot: str
    flange_link: str                      # link/body whose +z points out of the flange
    home: tuple                           # arm joint angles (rad), IK start seed
    rail: bool = False
    rail_range: float = 0.8               # +- m along world y
    pedestal: float = 0.0                 # base (or rail top) height, m
    panel_center: tuple = (0.70, 0.0, 0.55)


LAYOUTS: dict[str, CellLayout] = {
    # Panda (0.85 m reach): rail + 0.3 m pedestal needed for the 1.0 m curved door
    "panda": CellLayout("panda", "attachment", (0, 0, 0, -1.57079, 0, 1.57079, -0.7853),
                        rail=True, rail_range=0.8, pedestal=0.3,
                        panel_center=(0.70, 0.0, 0.55)),
    # FANUC CRX-10iA/L (1.42 m reach; FANUC sells a paint variant). Fixed, it can't
    # reach the curved door's top corners with the gun hanging under the flange.
    "crx10ia_l": CellLayout("crx10ia_l", "fanuc_flange", (0, 0, 0, 0, -np.pi / 2, 0),
                            rail=True, rail_range=0.8, pedestal=0.06,
                            panel_center=(0.80, 0.0, 0.60)),
}


DOOR_WINDOW = (-0.35, 0.08, 0.35, 0.26)
PANEL_KINDS = ("flat", "door", "curved")


def make_panel(kind: str):
    """The three reference panels: flat, door (flat + window), curved (R = 0.67 m)."""
    from .panel import Panel
    if kind not in PANEL_KINDS:
        raise ValueError(f"panel kind must be one of {PANEL_KINDS}")
    if kind == "curved":
        return Panel(curvature=1.5)
    p = Panel()
    if kind == "door":
        p.add_cutout(*DOOR_WINDOW)
    return p


def _mat2quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


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
        """(w, x, y, z)"""
        return _mat2quat_wxyz(self.R)
