"""Scripted-expert trajectories for Isaac Sim, precomputed with the MuJoCo cell.

The raster planner + IK (mujoco_cell) produce a joint-space path that was verified
locally (tracking, coverage, validation). This module saves that path, plus
everything Isaac needs to rebuild the same episode, to one .npz file. The Isaac
env then just plays the joint targets back with its own physics, so no IK code
runs inside Isaac.

    python scripts/isaac/export_expert.py --panel door      # -> outputs/expert_crx10ia_l_door_full.npz

Exporting needs mujoco (pip install mujoco, also fine on the Linux cloud box);
loading does not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXPERT_VERSION = 1


def export_expert(panel_kind: str, robot: str = "crx10ia_l", region: str = "full",
                  out: str | Path | None = None, target_um: float = 50.0) -> Path:
    from ..cell_layout import make_panel
    from ..mujoco_cell.cell import PaintCell
    from ..mujoco_cell.scene import CONTROL_HZ

    panel = make_panel(panel_kind)
    target = panel.region_mask(region)
    cell = PaintCell(panel, target, mode="kinematic", robot=robot, target_um=target_um)
    S, m = cell.schedule, cell.model
    reference = {k: (float(v) if not isinstance(v, bool) else v)
                 for k, v in cell.run().report().items()}   # what Isaac should reproduce
    # joint names as the URDF has them ("rail", "J1".."J6"), in qpos order
    names = [m.joint(i).name.split("/")[-1] for i in range(m.njnt)]
    meta = dict(version=EXPERT_VERSION, robot=robot, panel_kind=panel_kind, region=region,
                control_hz=CONTROL_HZ, target_um=target_um, n_lead=int(S.n_lead),
                joint_names=names, mujoco_reference=reference,
                panel=dict(width=panel.width, height=panel.height, res=panel.res,
                           curvature=panel.curvature, cutouts=panel.cutouts),
                panel_center=cell.pose.center.tolist(), panel_R=cell.pose.R.tolist(),
                gun=dict(cone_half_angle_deg=cell.gun.cone_half_angle_deg,
                         flow_rate=cell.gun.flow_rate,
                         optimal_distance=cell.gun.optimal_distance,
                         efficiency_width=cell.gun.efficiency_width))
    out = Path(out or Path("outputs") / f"expert_{robot}_{panel_kind}_{region.replace(' ', '_')}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, q=S.q.astype(np.float32), qd=S.qd.astype(np.float32),
                        spray=S.spray.astype(np.float32),
                        nozzle_pos=S.pos.astype(np.float32), nozzle_dir=S.dir.astype(np.float32),
                        target=target, meta=json.dumps(meta))
    return out


def load_expert(path: str | Path) -> dict:
    """Arrays plus `meta` (dict), and `panel` / `pose` rebuilt from it."""
    from ..cell_layout import PanelPose
    from ..panel import Panel

    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("version") != EXPERT_VERSION:
        raise ValueError(f"{path}: expert file version {meta.get('version')}, "
                         f"expected {EXPERT_VERSION}; re-export it")
    p = meta["panel"]
    panel = Panel(p["width"], p["height"], p["res"], p["curvature"])
    for c in p["cutouts"]:
        panel.add_cutout(*c)
    out = {k: z[k] for k in ("q", "qd", "spray", "nozzle_pos", "nozzle_dir", "target")}
    out.update(meta=meta, panel=panel,
               pose=PanelPose(center=np.array(meta["panel_center"]), R=np.array(meta["panel_R"])))
    return out
