"""Isaac Lab 3.0 painting environment (direct workflow). Runs on the cloud GPU.

STATUS: written against the Isaac Lab 3.0 docs but not yet run (Isaac Sim can't run
on the dev laptop). Where 3.0 renamed an API, this uses the 2.x name that 3.0 still
accepts with a deprecation warning, through the small helpers below, so a rename is
a one-line fix. Everything it loads (robot URDF, expert trajectory, panel USD) is
tested locally by tests/test_isaac_task.py.

Per env: the rail-mounted FANUC CRX + spray gun (one URDF, robot_urdf.py), a panel
(panel_usd.py) at the same pose as the MuJoCo cell. All envs share one panel type
per run. Each 30 Hz step:
  1. joint position targets: the precomputed scripted expert (expert.py), or a
     policy's actions (7 joint deltas + spray on/off)
  2. 20 physics steps at 600 Hz (decimation 20)
  3. nozzle pose from the `nozzle` link -> panel frame -> BatchedPaintSim on the GPU
     for all envs at once (the Phase 2a simulator)
  4. env 0's coat is copied to the panel's vertex colours every `viz_every` steps
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from ..torch_sim import BatchedPaintSim
from .expert import load_expert
from .panel_usd import panel_mesh, set_vertex_colors, vertex_colors

CONTROL_HZ = 30
DECIMATION = 20
# same servo gains as the verified MuJoCo cell (mujoco_cell/robots.py)
ARM_KP = {"J1": 8000.0, "J2": 8000.0, "J3": 5000.0, "J4": 1500.0, "J5": 1500.0, "J6": 800.0}
ARM_KV = {"J1": 800.0, "J2": 800.0, "J3": 500.0, "J4": 150.0, "J5": 150.0, "J6": 80.0}


# ------------------------------------------------------- 3.0 compatibility helpers
def _t(x) -> torch.Tensor:
    """3.0 data properties return ProxyArray (use .torch); 2.x returned tensors."""
    return x.torch if hasattr(x, "torch") else x


def _xyzw(q_wxyz) -> tuple:
    """Isaac Lab 3.0 quaternions are (x, y, z, w)."""
    w, x, y, z = (float(v) for v in q_wxyz)
    return (x, y, z, w)


def _set_position_targets(robot, q, joint_ids):
    if hasattr(robot, "set_joint_position_target"):          # 2.x name, deprecated in 3.0
        robot.set_joint_position_target(q, joint_ids=joint_ids)
    else:                                                     # 3.1+
        robot.actuators.target_command.set_position_index(position=q, joint_ids=joint_ids)


def _write_joint_state(robot, q, qd, joint_ids, env_ids):
    if hasattr(robot, "write_joint_position_to_sim_index"):  # 3.0
        robot.write_joint_position_to_sim_index(position=q, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=qd, joint_ids=joint_ids, env_ids=env_ids)
    else:                                                     # 2.x
        robot.write_joint_state_to_sim(q, qd, joint_ids=joint_ids, env_ids=env_ids)


def _stage():
    try:
        return sim_utils.get_current_stage()
    except AttributeError:
        import omni.usd
        return omni.usd.get_context().get_stage()


# -------------------------------------------------------------------------- config
@configclass
class PaintSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)))
    robot: ArticulationCfg = None          # filled by make_env_cfg()
    panel: AssetBaseCfg = None


@configclass
class PaintEnvCfg(DirectRLEnvCfg):
    decimation = DECIMATION
    episode_length_s = 60.0                # replaced by the expert's length
    action_space = 8                       # 7 joint deltas (rail + J1..J6) + spray
    observation_space = 22
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / (CONTROL_HZ * DECIMATION),
                                       render_interval=DECIMATION)
    scene: PaintSceneCfg = PaintSceneCfg(num_envs=16, env_spacing=3.0, replicate_physics=True)
    expert_path: str = ""
    scripted: bool = True                  # play the expert; False = use actions
    action_scale: float = 0.05             # rad (or m) per step for policy actions
    viz_every: int = 3                     # update env 0's paint colours every N steps


def make_env_cfg(expert_path: str, robot_usd: str, panel_usd: str, num_envs: int = 16,
                 scripted: bool = True) -> PaintEnvCfg:
    e = load_expert(expert_path)
    cfg = PaintEnvCfg()
    cfg.expert_path, cfg.scripted = str(expert_path), scripted
    cfg.episode_length_s = len(e["spray"]) / CONTROL_HZ
    cfg.scene.num_envs = num_envs
    q0 = dict(zip(e["meta"]["joint_names"], map(float, e["q"][0])))
    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd,
            # gravity off on the arm = the MuJoCo cell's gravity compensation
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=q0),
        actuators={
            "rail": ImplicitActuatorCfg(joint_names_expr=["rail"], stiffness=20000.0,
                                        damping=1500.0),
            "arm": ImplicitActuatorCfg(joint_names_expr=["J[1-6]"], stiffness=ARM_KP,
                                       damping=ARM_KV),
        })
    pose = e["pose"]
    cfg.scene.panel = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Panel",
        spawn=sim_utils.UsdFileCfg(usd_path=panel_usd),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(map(float, pose.center)),
                                                rot=_xyzw(pose.quat)))
    return cfg


# ----------------------------------------------------------------------------- env
class PaintEnv(DirectRLEnv):
    cfg: PaintEnvCfg

    def __init__(self, cfg: PaintEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        e = load_expert(cfg.expert_path)
        self.meta, self.panel = e["meta"], e["panel"]
        dev, n = self.device, self.num_envs

        ids, names = self.robot.find_joints(self.meta["joint_names"], preserve_order=True)
        assert list(names) == list(self.meta["joint_names"]), (names, self.meta["joint_names"])
        self.joint_ids = ids
        self.nozzle_id = self.robot.find_bodies("nozzle")[0][0]

        self.q_exp = torch.as_tensor(e["q"], device=dev)            # (T, 7)
        self.qd_exp = torch.as_tensor(e["qd"], device=dev)
        self.spray_exp = torch.as_tensor(e["spray"], device=dev)    # (T,)
        self.T = len(e["spray"])
        self.panel_c = torch.as_tensor(e["pose"].center, device=dev, dtype=torch.float32)
        self.panel_R = torch.as_tensor(e["pose"].R, device=dev, dtype=torch.float32)
        self.z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)

        p, g = self.panel, self.meta["gun"]
        self.paint = BatchedPaintSim(n, p.width, p.height, p.res, device=dev,
                                     window_half_m=0.2)
        self.paint.set_panel(None, curvature=p.curvature, cutouts=p.cutouts)
        self.paint.set_target(None, torch.as_tensor(e["target"]))
        self.paint.set_gun(None, **g)
        self.target_um = self.meta["target_um"]
        self._q_target = self.q_exp[0].repeat(n, 1)
        self._spray = torch.zeros(n, device=dev)
        self._last_cov = torch.zeros(n, device=dev)
        self.last_metrics = None

        _, _, self._vert_cells = panel_mesh(p)
        self._viz_prim = _stage().GetPrimAtPath("/World/envs/env_0/Panel/surface")
        if not self._viz_prim.IsValid():
            print("[paintbot] env_0 panel mesh not found; live paint colours disabled")
            self._viz_prim = None

    # --------------------------------------------------------------- stepping
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        t = self.episode_length_buf.clamp(max=self.T - 1)
        if self.cfg.scripted:
            self._q_target = self.q_exp[t]
            self._spray = self.spray_exp[t]
        else:
            q = _t(self.robot.data.joint_pos)[:, self.joint_ids]
            self._q_target = q + self.cfg.action_scale * actions[:, :7].clamp(-1, 1)
            self._spray = (actions[:, 7] > 0).float()

    def _apply_action(self) -> None:
        _set_position_targets(self.robot, self._q_target, self.joint_ids)

    def _nozzle_in_panel_frame(self):
        pos = _t(self.robot.data.body_pos_w)[:, self.nozzle_id] - self.scene.env_origins
        quat = _t(self.robot.data.body_quat_w)[:, self.nozzle_id]
        axis = math_utils.quat_apply(quat, self.z_axis)
        return (pos - self.panel_c) @ self.panel_R, axis @ self.panel_R

    def _paint_step(self):
        p, d = self._nozzle_in_panel_frame()
        self.paint.step(p, d, self._spray, dt=1.0 / CONTROL_HZ)
        if self._viz_prim is not None and int(self.common_step_counter) % self.cfg.viz_every == 0:
            th = self.paint.thickness[0].detach().cpu().numpy()
            set_vertex_colors(self._viz_prim, vertex_colors(th, self.panel, self._vert_cells,
                                                            self.target_um))

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._paint_step()                       # first hook after the physics steps
        time_out = self.episode_length_buf >= self.T - 1
        return torch.zeros_like(time_out), time_out

    def _get_rewards(self) -> torch.Tensor:
        cov = self.paint.metrics(self.target_um)["in_spec"]
        r = (cov - self._last_cov) * 100.0
        self._last_cov = cov
        return r

    def _get_observations(self) -> dict:
        q = _t(self.robot.data.joint_pos)[:, self.joint_ids]
        qd = _t(self.robot.data.joint_vel)[:, self.joint_ids]
        p, d = self._nozzle_in_panel_frame()
        prog = (self.episode_length_buf.float() / self.T).unsqueeze(1)
        obs = torch.cat([q, qd, p, d, prog, self._last_cov.unsqueeze(1)], dim=1)
        return {"policy": obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # keep the finished episodes' metrics before clearing the paint
        m = self.paint.metrics(self.target_um)
        self.last_metrics = {k: v.detach().cpu().numpy() for k, v in m.items()}
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        q0 = self.q_exp[0].repeat(len(ids), 1)
        _write_joint_state(self.robot, q0, torch.zeros_like(q0), self.joint_ids, ids)
        self.paint.reset(ids)
        self._last_cov[ids] = 0.0

    # ----------------------------------------------------------------- metrics
    def metrics_now(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in self.paint.metrics(self.target_um).items()}
