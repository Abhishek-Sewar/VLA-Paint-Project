"""Per-episode validation: decides whether a simulated demo is clean enough to train on.

Ported from the original project's four checks, with three fixes:
  * the spray-miss check uses where the spray *axis hits the panel*, and only
    while spraying (the original compared the end-effector position itself,
    which is never on the panel; it is 25 cm in front of it);
  * camera checks run on every frame, vectorised, not just one;
  * every failure reason is returned, not only the first, so the morning summary
    shows co-occurring problems.

Everything is array arithmetic, so it costs a few milliseconds per episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ValidationConfig:
    joint_limits: np.ndarray | None = None   # (J, 2) radians; None skips the check
    min_mean_pixel: float = 10.0             # below this a frame counts as black
    max_offpanel_spray_frac: float = 0.20    # fraction of spray-on steps aimed off-panel
    min_coverage: float = 0.80               # client spec


@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_episode(ep: dict, cfg: ValidationConfig) -> ValidationResult:
    """`ep` keys (all optional except where noted):

    collision     (T,) bool       contact between robot and panel/fixtures
    joint_pos     (T, J) float    radians
    rgb           (T, H, W, 3) uint8
    depth         (T, H, W) float
    spray         (T,) float      required for the spray-miss check
    aim_on_panel  (T,) bool       spray axis hits a paintable cell
    coverage      float           from coverage_report()
    """
    reasons, stats = [], {}

    if "collision" in ep and np.any(ep["collision"]):
        reasons.append("collision")
        stats["first_collision_step"] = int(np.argmax(ep["collision"]))

    if cfg.joint_limits is not None and "joint_pos" in ep:
        q = np.asarray(ep["joint_pos"])
        lim = np.asarray(cfg.joint_limits)
        bad = (q < lim[:, 0]) | (q > lim[:, 1]) | ~np.isfinite(q)
        if bad.any():
            reasons.append("joint_limit")
            stats["joint_violations"] = int(bad.sum())

    if "rgb" in ep:
        means = np.asarray(ep["rgb"]).reshape(len(ep["rgb"]), -1).mean(axis=1)
        if np.any(means < cfg.min_mean_pixel):
            reasons.append("black_frame")
            stats["black_frames"] = int(np.sum(means < cfg.min_mean_pixel))

    if "depth" in ep and not np.all(np.isfinite(ep["depth"])):
        reasons.append("depth_nan")

    if "spray" in ep and "aim_on_panel" in ep:
        on = np.asarray(ep["spray"]) > 0
        if on.any():
            off = float(np.mean(~np.asarray(ep["aim_on_panel"])[on]))
            stats["offpanel_spray_frac"] = off
            if off > cfg.max_offpanel_spray_frac:
                reasons.append("spray_miss")

    if "coverage" in ep and ep["coverage"] < cfg.min_coverage:
        reasons.append("low_coverage")

    return ValidationResult(passed=not reasons, reasons=reasons, stats=stats)
