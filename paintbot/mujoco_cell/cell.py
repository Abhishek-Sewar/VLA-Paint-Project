"""PaintCell: the Panda follows the raster plan and paints the panel.

Per 30 Hz control tick:
  1. move the arm toward this tick's IK solution
       kinematic: set the joint angles directly (perfect tracking)
       dynamic:   drive the Panda's position actuators for 20 physics substeps
  2. read where the nozzle actually is, convert to panel coordinates
  3. BatchedPaintSim.step() lays down paint from that pose (dt = 1/30 s)
  4. log joints, tracking error, panel contact and whether the spray axis is on
     a paintable cell, for validate_episode()

The episode is: approach from a park pose with the spray off, settle, then the
planner's trajectory. Painting uses the measured nozzle pose, so in dynamic mode
tracking lag shows up in the coat.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from ..metrics import coverage_report
from ..panel import Panel
from ..planner import RasterPlanner, Trajectory
from ..spray import SprayGun
from ..torch_sim import BatchedPaintSim
from ..validation import ValidationConfig, ValidationResult, validate_episode
from .ik import NozzleIK
from .scene import (CONTROL_HZ, NOZZLE_SITE, PANEL_COLLISION, PANEL_TEX, SUBSTEPS,
                    PanelPose, build_scene)
from .texture import thickness_rgb, write_texture

APPROACH_S = 2.0
PARK_STANDOFF = 0.45
SETTLE_S = 0.5


@dataclass
class Schedule:
    q: np.ndarray            # (T, nq) commanded joint angles
    qd: np.ndarray           # (T, nq) commanded joint velocities (for feedforward)
    spray: np.ndarray        # (T,)
    pos: np.ndarray          # (T, 3) desired nozzle position, world
    dir: np.ndarray          # (T, 3) desired spray direction, world
    ik_pos_err: np.ndarray   # (T,)
    ik_ang_err: np.ndarray   # (T,)
    n_lead: int              # ticks of approach + settle before the plan starts


class PaintCell:
    def __init__(self, panel: Panel, target_mask: np.ndarray | None = None,
                 mode: str = "dynamic", target_um: float = 50.0,
                 gun: SprayGun | None = None, pose: PanelPose | None = None,
                 dtype: torch.dtype = torch.float32, device: str = "cpu"):
        if mode not in ("dynamic", "kinematic"):
            raise ValueError("mode must be 'dynamic' or 'kinematic'")
        self.panel, self.mode, self.target_um = panel, mode, float(target_um)
        self.gun = gun or SprayGun()
        self.target = panel.region_mask("full") if target_mask is None else target_mask

        self.model, self.pose = build_scene(panel, pose)
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(NOZZLE_SITE).id
        self.tex_id = self.model.texture(PANEL_TEX).id
        self.panel_geom = self.model.geom(PANEL_COLLISION).id
        self.ik = NozzleIK(self.model)

        self.traj: Trajectory = RasterPlanner(self.gun, target_thickness=self.target_um).plan(
            panel, self.target)
        self.schedule = self._make_schedule()

        self.sim = BatchedPaintSim(1, panel.width, panel.height, panel.res,
                                   device=device, dtype=dtype)
        self.sim.set_panel([0], curvature=panel.curvature, cutouts=panel.cutouts)
        self.sim.set_target([0], self.target)
        self.sim.set_gun([0], cone_half_angle_deg=self.gun.cone_half_angle_deg,
                         flow_rate=self.gun.flow_rate,
                         optimal_distance=self.gun.optimal_distance,
                         efficiency_width=self.gun.efficiency_width)
        self._ctrl_map()
        self.reset()

    # -------------------------------------------------------------------- setup
    def _make_schedule(self) -> Schedule:
        tr, pose, ik = self.traj, self.pose, self.ik
        pos_w = pose.to_world(tr.pos)
        dir_w = pose.dir_to_world(tr.dir)
        # posture reference: nozzle square to the panel centre at standoff.
        # start pose ("park"): same, but 0.45 m back. The Panda's own `home` pose
        # puts the sideways gun inside the panel, so episodes start from park.
        n_c = pose.dir_to_world(np.array([0.0, 0.0, 1.0]))
        c = pose.center
        q_ref, _, _ = ik.solve(c + n_c * self.gun.optimal_distance, -n_c)
        ik.set_posture(q_ref)
        self.q_park, _, _ = ik.solve(c + n_c * PARK_STANDOFF, -n_c, q_ref)
        home = self.q_park

        q0, _, _ = ik.solve(pos_w[0], dir_w[0], q_ref)
        res = ik.solve_trajectory(pos_w, dir_w, q0=q0)

        n_app = int(APPROACH_S * CONTROL_HZ)
        n_set = int(SETTLE_S * CONTROL_HZ)
        s = np.linspace(0.0, 1.0, n_app)
        s = s * s * (3 - 2 * s)                                  # smoothstep
        q_app = home + s[:, None] * (res.q[0] - home)
        q = np.concatenate([q_app, np.repeat(res.q[:1], n_set, 0), res.q])
        lead = n_app + n_set

        # desired nozzle pose for every tick (approach: wherever FK puts it)
        lead_pos, lead_dir = zip(*(ik.nozzle_pose(qq) for qq in q[:lead]))
        qd = np.gradient(q, 1.0 / CONTROL_HZ, axis=0)
        return Schedule(
            q=q, qd=qd,
            spray=np.concatenate([np.zeros(lead), tr.spray]),
            pos=np.concatenate([np.array(lead_pos), pos_w]),
            dir=np.concatenate([np.array(lead_dir), dir_w]),
            ik_pos_err=np.concatenate([np.zeros(lead), res.pos_err]),
            ik_ang_err=np.concatenate([np.zeros(lead), res.ang_err]),
            n_lead=lead)

    def _ctrl_map(self):
        """actuator -> qpos index, and the kv/kp ratio used for velocity feedforward.

        The Panda's actuators apply kp*(ctrl - q) - kv*qdot. Commanding
        ctrl = q_des + (kv/kp)*qd_des turns that into kp*(q_des - q) + kv*(qd_des - qdot),
        which removes the steady lag of ~0.1 s that pure position targets would have.
        """
        m = self.model
        self._act_q = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        kp = m.actuator_gainprm[:, 0]
        kv = -m.actuator_biasprm[:, 2]
        self._ff = np.where(kp > 0, kv / np.where(kp > 0, kp, 1.0), 0.0)

    # ------------------------------------------------------------------ running
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.q_park
        self.data.ctrl[:] = self.q_park[self._act_q]
        mujoco.mj_forward(self.model, self.data)
        self.sim.reset()
        self.t = 0
        T = len(self.schedule.spray)
        self.log = dict(
            joint_pos=np.zeros((T, self.model.nq)),
            nozzle_pos=np.zeros((T, 3)), nozzle_dir=np.zeros((T, 3)),
            pos_err=np.zeros(T), ang_err=np.zeros(T),
            collision=np.zeros(T, dtype=bool), aim_on_panel=np.zeros(T, dtype=bool))
        self.update_texture()

    @property
    def done(self) -> bool:
        return self.t >= len(self.schedule.spray)

    def __len__(self) -> int:
        return len(self.schedule.spray)

    def nozzle_pose(self):
        """Current nozzle position and spray direction, world frame."""
        return (self.data.site_xpos[self.site].copy(),
                self.data.site_xmat[self.site].reshape(3, 3)[:, 2].copy())

    def step(self):
        """Advance one 30 Hz control tick. Returns the spray intensity used."""
        S, t, m, d = self.schedule, self.t, self.model, self.data
        if self.mode == "kinematic":
            d.qpos[:] = S.q[t]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
        else:
            q_prev = S.q[t - 1] if t > 0 else d.qpos.copy()
            for k in range(1, SUBSTEPS + 1):
                a = k / SUBSTEPS                         # interpolate within the tick
                q_des = q_prev + a * (S.q[t] - q_prev)
                d.ctrl[:] = q_des[self._act_q] + self._ff * S.qd[t][self._act_q]
                mujoco.mj_step(m, d)

        p, a = self.nozzle_pose()
        spray = float(S.spray[t])
        p_panel = self.pose.to_panel(p)
        d_panel = self.pose.dir_to_panel(a)
        f = dict(dtype=self.sim.dtype, device=self.sim.device)
        pp = torch.as_tensor(p_panel[None], **f)
        dd = torch.as_tensor(d_panel[None], **f)
        self.sim.step(pp, dd, torch.tensor([spray], **f), dt=1.0 / CONTROL_HZ)

        L = self.log
        L["joint_pos"][t] = d.qpos
        L["nozzle_pos"][t], L["nozzle_dir"][t] = p, a
        L["pos_err"][t] = np.linalg.norm(p - S.pos[t])
        L["ang_err"][t] = np.arccos(np.clip(a @ S.dir[t], -1.0, 1.0))
        L["collision"][t] = self._touching_panel()
        L["aim_on_panel"][t] = bool(self.sim.aim_on_panel(pp, dd)[0])
        self.t += 1
        return spray

    def run(self, callback=None):
        while not self.done:
            self.step()
            if callback is not None:
                callback(self)
        return self

    def _touching_panel(self) -> bool:
        con = self.data.contact[:self.data.ncon]
        g = self.panel_geom
        return bool(np.any((con.geom1 == g) | (con.geom2 == g))) if self.data.ncon else False

    # ----------------------------------------------------------------- outputs
    @property
    def thickness(self) -> np.ndarray:
        return self.sim.thickness[0].detach().cpu().double().numpy()

    def update_texture(self):
        rgb = thickness_rgb(self.thickness, self.panel.paintable, self.target_um)
        write_texture(self.model, self.tex_id, rgb)

    def report(self) -> dict:
        """coverage_report() for the paint laid down so far (plus transfer efficiency)."""
        return coverage_report(self.panel, self.thickness, self.target, self.target_um,
                               deposited=float(self.sim.deposited[0]),
                               emitted=float(self.sim.emitted[0]))

    def episode(self) -> dict:
        n = self.t
        return dict(collision=self.log["collision"][:n], joint_pos=self.log["joint_pos"][:n],
                    spray=self.schedule.spray[:n], aim_on_panel=self.log["aim_on_panel"][:n],
                    coverage=self.report()["coverage"])

    def validate(self, joint_margin: float = 0.01) -> ValidationResult:
        """validate_episode() with the model's joint limits (padded by joint_margin,
        since MuJoCo's soft limits allow tiny overshoot)."""
        lim = self.model.jnt_range.copy()
        lim[:, 0] -= joint_margin
        lim[:, 1] += joint_margin
        return validate_episode(self.episode(), ValidationConfig(joint_limits=lim))

    def tracking_stats(self) -> dict:
        """Max nozzle tracking error over spray-on ticks."""
        on = self.schedule.spray[:self.t] > 0
        pe, ae = self.log["pos_err"][:self.t][on], self.log["ang_err"][:self.t][on]
        return dict(max_pos_err_mm=float(pe.max() * 1000) if on.any() else 0.0,
                    mean_pos_err_mm=float(pe.mean() * 1000) if on.any() else 0.0,
                    max_ang_err_deg=float(np.degrees(ae.max())) if on.any() else 0.0,
                    max_ik_pos_err_mm=float(self.schedule.ik_pos_err.max() * 1000),
                    max_ik_ang_err_deg=float(np.degrees(self.schedule.ik_ang_err.max())))
