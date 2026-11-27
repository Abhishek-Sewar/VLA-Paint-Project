"""paintbot: simulator-agnostic core for a language-conditioned robotic spray-painting system.

Everything here is plain NumPy so it runs on any laptop. The same math gets ported
to batched GPU tensors when we move into Isaac Lab (Phase 2).
"""
from .panel import Panel
from .spray import SprayGun
from .planner import RasterPlanner, Trajectory
from .metrics import coverage_report
from .validation import ValidationConfig, validate_episode

__all__ = [
    "Panel", "SprayGun", "RasterPlanner", "Trajectory",
    "coverage_report", "ValidationConfig", "validate_episode",
]
