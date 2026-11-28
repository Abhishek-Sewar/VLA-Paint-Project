"""Panel as a USD mesh with per-vertex colours, so Isaac Sim shows the coat live.

Only Pixar's USD API (pxr) is used: it ships with Isaac Sim, and `pip install
usd-core` gives the same API on a laptop, which is how tests/test_isaac_task.py
checks this file without Isaac.

The mesh is authored in panel coordinates (panel.py), so the prim's transform is
the panel pose. Paint is shown by rewriting `primvars:displayColor` (one colour
per vertex, sampled from the thickness grid with paint_colors.thickness_rgb).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..paint_colors import thickness_rgb


def panel_mesh(panel, step: int = 2):
    """Vertices every `step` cells. Returns points (N,3) in panel frame,
    face indices (F,4) as quads, and each vertex's (iy, ix) grid cell."""
    nx = int(np.ceil(panel.nx / step)) + 1
    ny = int(np.ceil(panel.ny / step)) + 1
    us = np.linspace(-panel.width / 2, panel.width / 2, nx)
    vs = np.linspace(-panel.height / 2, panel.height / 2, ny)
    U, V = np.meshgrid(us, vs)
    P, _ = panel.surface(U, V)
    ix = np.clip(np.round((U + panel.width / 2) / panel.res - 0.5).astype(int), 0, panel.nx - 1)
    iy = np.clip(np.round((V + panel.height / 2) / panel.res - 0.5).astype(int), 0, panel.ny - 1)
    idx = np.arange(nx * ny).reshape(ny, nx)
    # counter-clockwise seen from the painted side (+z in panel frame)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    return P.reshape(-1, 3), quads, np.stack([iy.ravel(), ix.ravel()], 1)


def vertex_colors(thickness, panel, vert_cells, target_um: float = 50.0) -> np.ndarray:
    """(N, 3) float32 in 0..1 for the mesh vertices."""
    rgb = thickness_rgb(thickness, panel.paintable, target_um)
    return rgb[vert_cells[:, 0], vert_cells[:, 1]].astype(np.float32) / 255.0


def write_panel_usd(panel, path: str | Path, step: int = 2) -> Path:
    """Write a standalone panel USD (default prim /Panel, mesh /Panel/surface)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    pts, quads, cells = panel_mesh(panel, step)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Panel")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Panel/surface")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(quads.ravel().tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    lo, hi = pts.min(0), pts.max(0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    col = vertex_colors(np.zeros((panel.ny, panel.nx)), panel, cells)
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                 UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec3fArray.FromNumpy(col))
    # remember the vertex -> cell map so the runtime doesn't have to recompute it
    root.GetPrim().CreateAttribute("paintbot:vertCells", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(cells.ravel().tolist()))
    stage.GetRootLayer().Save()
    return path


def set_vertex_colors(mesh_prim, colors: np.ndarray) -> None:
    """Overwrite displayColor on a live mesh prim (e.g. inside Isaac Sim)."""
    from pxr import UsdGeom, Vt
    UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("displayColor").Set(
        Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors, dtype=np.float32)))
