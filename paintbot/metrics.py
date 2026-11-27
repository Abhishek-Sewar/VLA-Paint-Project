"""Paint-quality metrics.

The original project used one number, "coverage". A real paint shop cares about
more than that, and splitting it apart is what lets failure analysis say *why*
an episode failed (thin coat vs. pooling vs. overspray onto glass).
"""
from __future__ import annotations

import numpy as np

from .panel import Panel


def coverage_report(panel: Panel, thickness: np.ndarray, target_mask: np.ndarray,
                    target_thickness: float, *, lo: float = 0.7, hi: float = 1.5,
                    overspray_frac: float = 0.2, glass_frac: float = 0.2,
                    deposited: float | None = None, emitted: float | None = None,
                    min_coverage: float = 0.80) -> dict:
    T = target_thickness
    target = target_mask & panel.paintable
    t = thickness[target]

    nontarget = panel.paintable & ~target_mask
    glass = ~panel.paintable

    rep = {
        "coverage": float(np.mean(t >= lo * T)),                       # thick enough
        "in_spec": float(np.mean((t >= lo * T) & (t <= hi * T))),      # thick enough, not pooled
        "pooling": float(np.mean(t > hi * T)),                         # too thick: runs / sags
        "mean_rel_thickness": float(t.mean() / T),
        "uniformity_cv": float(t.std() / max(t.mean(), 1e-9)),
        "overspray_nontarget": float(np.mean(thickness[nontarget] > overspray_frac * T))
        if nontarget.any() else 0.0,
        "glass_contamination": float(np.mean(thickness[glass] > glass_frac * T))
        if glass.any() else 0.0,
    }
    if deposited is not None and emitted:
        rep["transfer_efficiency"] = float(deposited / emitted)
    rep["accept"] = rep["coverage"] >= min_coverage
    return rep
