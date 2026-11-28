"""Inverse kinematics for the spray nozzle: task-priority damped least squares.

  1. primary (always met): nozzle position + spray-axis direction (5 DoF)
  2. secondary: roll about the spray axis such that the nozzle y axis (= flange
     +z) points down, i.e. the gun hangs below a downward-pointing flange like on
     a paint robot; yawing the spray across a curved panel is then joint 7.
     It is met wherever the spare DoF allow and gives way near the workspace
     edge (top corners of the curved door, bottom rows where the elbow folds).
  3. tertiary: a reference posture, with the rail carriage under the wrist.

Why the roll preference: with roll simply free, the arm drifted between wrist
configurations along a pass and then had to jump. As a hard constraint it put
the curved door's top corners out of reach. Joints pinned at a limit drop out of
the Jacobian (active set).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import NOZZLE_SITE

WRIST_BACK = 0.3          # metres from nozzle back to the wrist, along the spray axis
_DOWN = np.array([0.0, 0.0, -1.0])


@dataclass
class IKResult:
    q: np.ndarray            # (T, nq) joint angles
    pos_err: np.ndarray      # (T,) metres
    ang_err: np.ndarray      # (T,) radians, spray axis vs target direction


def nozzle_frame(d) -> np.ndarray:
    """Target nozzle rotation (columns x, y, z) for spray direction d."""
    z = np.asarray(d, float) / np.linalg.norm(d)
    y = _DOWN - (_DOWN @ z) * z
    if np.linalg.norm(y) < 1e-6:                  # spraying straight down/up
        y = np.array([1.0, 0.0, 0.0]) - z[0] * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


class NozzleIK:
    def __init__(self, model: mujoco.MjModel, home=None, damping: float = 1e-3,
                 posture_gain: float = 0.1, max_iters: int = 200, min_iters: int = 3,
                 pos_tol: float = 1e-5, ang_tol: float = 1e-4):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(NOZZLE_SITE).id
        self.nv = model.nv
        self.lo = model.jnt_range[:, 0].copy()
        self.hi = model.jnt_range[:, 1].copy()
        if home is None:
            home = model.key("home").qpos if model.nkey else np.zeros(model.nq)
        self.ref = np.asarray(home, float).copy()  # preferred posture (see set_posture)
        self.lam2 = damping ** 2
        self.k_null = posture_gain
        self.max_iters, self.pos_tol, self.ang_tol = max_iters, pos_tol, ang_tol
        # keep iterating a little after convergence so the posture term acts every
        # tick; otherwise warm-started solves stop at once and the spare DoF drift
        self.min_iters = min_iters
        self._jp = np.zeros((3, self.nv))
        self._jr = np.zeros((3, self.nv))
        rail = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rail")
        self.rail_adr = int(model.jnt_qposadr[rail]) if rail >= 0 else None

    def set_posture(self, q):
        """Posture the null space pulls toward (the rail entry is overridden per target)."""
        self.ref = np.asarray(q, float).copy()

    def posture(self, p_target, d_target):
        """Reference posture with the rail carriage under the wrist (about 0.3 m
        behind the nozzle; on the curved door's edges ~0.2 m further out than it)."""
        q = self.ref.copy()
        if self.rail_adr is not None:
            i = self.rail_adr
            q[i] = np.clip(p_target[1] - WRIST_BACK * d_target[1], self.lo[i], self.hi[i])
        return q

    def nozzle_pose(self, q):
        """(position, spray direction) of the nozzle for joint angles q."""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        return (self.d.site_xpos[self.site].copy(),
                self.d.site_xmat[self.site].reshape(3, 3)[:, 2].copy())

    def errors(self, q, p_target, d_target):
        p, a = self.nozzle_pose(q)
        d_target = np.asarray(d_target, float) / np.linalg.norm(d_target)
        return (float(np.linalg.norm(p_target - p)),
                float(np.arccos(np.clip(a @ d_target, -1.0, 1.0))))

    def solve(self, p_target, d_target, q0=None, restarts: int = 8, seed: int = 0):
        """Returns (q, pos_err, ang_err) for one nozzle target (world frame).

        Starts from q0 (or the preferred posture). If that ends in a local minimum,
        retries from random postures and keeps the best.
        """
        best = self._solve_from(p_target, d_target, q0)
        rng = np.random.default_rng(seed)
        for _ in range(restarts):
            if best[1] < 1e-3 and best[2] < np.radians(0.5):
                break
            q_seed = rng.uniform(self.lo, self.hi) * 0.8 + self.posture(p_target, d_target) * 0.2
            cand = self._solve_from(p_target, d_target, q_seed)
            if cand[1] + 0.1 * cand[2] < best[1] + 0.1 * best[2]:
                best = cand
        return best

    def _solve_from(self, p_target, d_target, q0=None):
        p_target = np.asarray(p_target, float)
        d_target = np.asarray(d_target, float) / np.linalg.norm(d_target)
        R_t = nozzle_frame(d_target)
        q_pref = self.posture(p_target, d_target)
        q = (q_pref if q0 is None else np.asarray(q0, float)).copy()
        I = np.eye(self.nv)
        for it in range(self.max_iters):
            self.d.qpos[:] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            R = self.d.site_xmat[self.site].reshape(3, 3)
            e_p = p_target - self.d.site_xpos[self.site]
            e_r = 0.5 * sum(np.cross(R[:, k], R_t[:, k]) for k in range(3))
            if R[:, 2] @ R_t[:, 2] < 0:                  # pointing backwards: big step
                e_r = np.cross(R[:, 2], R_t[:, 2] + 1e-3) * 10.0
            if (it >= self.min_iters and np.linalg.norm(e_p) < self.pos_tol
                    and np.linalg.norm(np.cross(R[:, 2], R_t[:, 2])) < self.ang_tol):
                break
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            a = R[:, 2]
            Pa = np.eye(3) - np.outer(a, a)
            # primary: position + spray axis (5 rows; roll about the axis removed)
            J1_full = np.vstack([self._jp, Pa @ self._jr])
            e1 = np.concatenate([e_p, Pa @ e_r])
            # secondary: roll about the spray axis toward "flange down"
            j2_full = a @ self._jr
            e2 = float(a @ e_r)
            # active set: a joint pinned at a limit and pushed further drops out of
            # the Jacobian, so the others take up the motion instead of the step
            # being clipped away
            free = np.ones(self.nv, dtype=bool)
            for _ in range(3):
                J1, j2 = J1_full * free, j2_full * free
                J1_pinv = J1.T @ np.linalg.inv(J1 @ J1.T + self.lam2 * np.eye(6))
                N1 = (I - J1_pinv @ J1) * free
                dq = J1_pinv @ e1
                J2 = j2 @ N1                               # roll, within primary null space
                s2 = J2 @ J2 + self.lam2
                dq = dq + J2 * ((e2 - j2 @ dq) / s2)
                N2 = N1 - np.outer(J2, J2) / s2
                dq = dq + N2 @ (self.k_null * (q_pref - q))
                pinned = free & (((q <= self.lo + 1e-6) & (dq < 0))
                                 | ((q >= self.hi - 1e-6) & (dq > 0)))
                if not pinned.any():
                    break
                free &= ~pinned
            n = np.linalg.norm(dq)
            if n > 0.2:                                  # limit step size
                dq *= 0.2 / n
            q = np.clip(q + dq, self.lo, self.hi)
        pe, ae = self.errors(q, p_target, d_target)
        return q, pe, ae

    def solve_trajectory(self, pos, dirs, q0=None, max_joint_step: float = 0.1) -> IKResult:
        """Warm-started IK along a path. pos, dirs: (T, 3) in world frame.

        No random restarts here: each step starts from the previous solution so the
        joint path stays continuous. Raises if any joint moves more than
        max_joint_step (rad or m) in one 30 Hz tick, i.e. the arm would have to jump.
        """
        T = len(pos)
        qs = np.zeros((T, self.m.nq))
        pe, ae = np.zeros(T), np.zeros(T)
        q = q0
        for t in range(T):
            q_prev = q
            q, pe[t], ae[t] = self.solve(pos[t], dirs[t], q, restarts=0)
            if q_prev is not None:
                jump = np.abs(q - q_prev).max()
                if jump > max_joint_step:
                    raise RuntimeError(f"IK jumped {jump:.3f} at step {t}/{T}; "
                                       "path leaves the arm's continuous workspace")
            qs[t] = q
        return IKResult(qs, pe, ae)
