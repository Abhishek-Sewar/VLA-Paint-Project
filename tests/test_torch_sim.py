"""Checks the batched PyTorch simulator against the NumPy reference (spray.py).

    python tests/test_torch_sim.py      # or: pytest -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
    try:
        import pytest
        pytest.skip("PyTorch not installed", allow_module_level=True)
    except ImportError:
        print("PyTorch not installed; skipping")
        sys.exit(0)

from paintbot import Panel, RasterPlanner, SprayGun, coverage_report  # noqa: E402
from paintbot.planner import simulate  # noqa: E402
from paintbot.torch_sim import BatchedPaintSim  # noqa: E402

CUT = (-0.35, 0.08, 0.35, 0.26)
# (curvature, cutouts, region) for three different envs running side by side
CONFIGS = [(0.0, [], "full"), (0.0, [CUT], "full"), (1.5, [], "upper right")]


def _reference():
    """NumPy panels, trajectories and final thickness for each config."""
    out = []
    for k, cuts, region in CONFIGS:
        p, g = Panel(curvature=k), SprayGun()
        for c in cuts:
            p.add_cutout(*c)
        tm = p.region_mask(region)
        tr = RasterPlanner(g).plan(p, tm)
        th, dep, emit = simulate(p, g, tr, window=False)  # exact reference
        out.append((p, tm, tr, th, dep, emit))
    return out


def _run_torch(ref, device="cpu", dtype=torch.float64, window_half_m=None):
    sim = BatchedPaintSim(len(ref), device=device, dtype=dtype, window_half_m=window_half_m)
    for i, ((k, cuts, _), (_, tm, *_)) in enumerate(zip(CONFIGS, ref)):
        sim.set_panel([i], curvature=k, cutouts=cuts)
        sim.set_target([i], tm)
    T = max(len(r[2]) for r in ref)
    pos = np.zeros((len(ref), T, 3))
    d = np.zeros((len(ref), T, 3))
    d[..., 2] = -1.0
    pos[..., 2] = 0.25
    spray = np.zeros((len(ref), T))
    for i, (_, _, tr, *_) in enumerate(ref):      # pad shorter episodes with spray off
        n = len(tr)
        pos[i, :n], d[i, :n], spray[i, :n] = tr.pos, tr.dir, tr.spray
    for t in range(T):
        sim.step(pos[:, t], d[:, t], spray[:, t], dt=1.0 / 30.0)
    return sim


def test_full_grid_matches_numpy():
    ref = _reference()
    sim = _run_torch(ref)
    for i, (_, _, _, th, dep, emit) in enumerate(ref):
        got = sim.thickness[i].cpu().numpy()
        err = np.abs(got - th).max() / th.max()
        assert err < 1e-6, (i, err)
        assert abs(sim.deposited[i].item() - dep) / dep < 1e-6
        assert abs(sim.emitted[i].item() - emit) / emit < 1e-9


def test_window_mode_close_to_full():
    ref = _reference()
    full = _run_torch(ref)
    win = _run_torch(ref, window_half_m=0.2)
    rel = ((win.thickness - full.thickness).abs().sum(dim=(1, 2))
           / full.thickness.sum(dim=(1, 2)))
    assert rel.max().item() < 0.005, rel


def test_metrics_match_numpy():
    ref = _reference()
    sim = _run_torch(ref)
    m = sim.metrics(50.0)
    for i, (p, tm, _, th, dep, emit) in enumerate(ref):
        r = coverage_report(p, th, tm, 50.0, deposited=dep, emitted=emit)
        for key in ("coverage", "in_spec", "pooling", "mean_rel_thickness",
                    "uniformity_cv", "overspray_nontarget", "glass_contamination",
                    "transfer_efficiency"):
            assert abs(m[key][i].item() - r[key]) < 1e-6, (i, key, m[key][i].item(), r[key])
        assert bool(m["accept"][i]) == r["accept"]


def test_float32_is_close_enough():
    ref = _reference()
    a = _run_torch(ref, dtype=torch.float64)
    b = _run_torch(ref, dtype=torch.float32)
    rel = (a.thickness - b.thickness.double()).abs().max() / a.thickness.max()
    assert rel.item() < 1e-4, rel.item()


def test_domain_randomized_flow_scales_paint():
    sim = BatchedPaintSim(3)
    sim.set_gun(flow_rate=torch.tensor([0.5, 0.75, 1.0], dtype=sim.dtype))
    pos = torch.tensor([[0.0, 0.0, 0.25]]).repeat(3, 1)
    d = torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)
    for _ in range(10):
        sim.step(pos, d, torch.ones(3))
    s = sim.thickness.sum(dim=(1, 2))
    assert abs((s[2] / s[0]).item() - 2.0) < 1e-4


def test_reset_clears_only_selected_envs():
    sim = BatchedPaintSim(2)
    pos = torch.tensor([[0.0, 0.0, 0.25]]).repeat(2, 1)
    d = torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)
    sim.step(pos, d, torch.ones(2))
    sim.reset([0])
    assert sim.thickness[0].sum().item() == 0.0
    assert sim.thickness[1].sum().item() > 0.0


def test_gpu_matches_cpu():
    if not torch.cuda.is_available():
        print("   (no CUDA GPU visible; skipped GPU check)")
        return
    ref = _reference()
    cpu = _run_torch(ref, dtype=torch.float32)
    gpu = _run_torch(ref, device="cuda", dtype=torch.float32)
    rel = (cpu.thickness - gpu.thickness.cpu()).abs().max() / cpu.thickness.max()
    assert rel.item() < 1e-4, rel.item()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
