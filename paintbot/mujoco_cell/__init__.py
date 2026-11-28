"""MuJoCo painting cell (Phase 2b-local): a robot arm on a rail + spray gun + panel.

Needs `pip install mujoco` and the robot models (python scripts/fetch_menagerie.py).
The core paintbot package never imports this one. Imports here are lazy, so the
mujoco-free submodule `assets` can be used without mujoco installed (the Isaac
side does that on the cloud box).
"""
__all__ = ["PaintCell", "PanelPose", "build_scene", "robots"]


def __getattr__(name):
    import importlib
    if name in ("robots", "assets", "scene", "cell", "ik", "texture"):
        return importlib.import_module(f"{__name__}.{name}")
    if name == "PaintCell":
        return importlib.import_module(f"{__name__}.cell").PaintCell
    if name == "build_scene":
        return importlib.import_module(f"{__name__}.scene").build_scene
    if name == "PanelPose":
        from ..cell_layout import PanelPose
        return PanelPose
    raise AttributeError(name)
