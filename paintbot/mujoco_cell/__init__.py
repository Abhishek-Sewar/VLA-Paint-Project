"""MuJoCo painting cell (Phase 2b-local): Franka Panda on a rail + spray gun + panel.

Needs `pip install mujoco` and the Menagerie Panda (python scripts/fetch_menagerie.py).
The core paintbot package never imports this one.
"""
from .cell import PaintCell
from .scene import PanelPose, build_scene

__all__ = ["PaintCell", "PanelPose", "build_scene"]
