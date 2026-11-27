"""How many env-steps per second can your machine paint?

    python scripts/bench_torch.py            # uses the GPU if PyTorch can see it
    python scripts/bench_torch.py --cpu      # force CPU
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paintbot.torch_sim import BatchedPaintSim  # noqa: E402


def bench(num_envs, device, window, steps=30):
    sim = BatchedPaintSim(num_envs, device=device, window_half_m=window)
    half = num_envs // 2                       # mix of flat and curved panels
    if half:
        sim.set_panel(list(range(half)), curvature=1.5)
    g = torch.Generator(device="cpu").manual_seed(0)
    u0 = torch.rand(num_envs, generator=g) * 0.8 - 0.4
    v0 = torch.rand(num_envs, generator=g) * 0.4 - 0.2
    d = torch.tensor([0.0, 0.0, -1.0]).repeat(num_envs, 1).to(device)
    on = torch.ones(num_envs, device=device)

    def poses(t):                                   # horizontal sweep, 0.3 m/s
        u = (u0 + 0.01 * t).clamp(-0.45, 0.45)
        return torch.stack([u, v0, torch.full_like(u, 0.25)], dim=-1).to(device)

    for t in range(5):                              # warm-up
        sim.step(poses(t), d, on)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(steps):
        sim.step(poses(t), d, on)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return num_envs * steps / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {name}\n")
    print(f"{'envs':>6} {'mode':>10} {'env-steps/s':>14} {'x real time':>12}")
    for window in (0.2, None):
        mode = "window" if window else "full grid"
        for n in (1, 16, 64, 256, 1024):
            try:
                sps = bench(n, device, window)
                print(f"{n:>6} {mode:>10} {sps:>14,.0f} {sps / 30:>12,.0f}")
            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError):
                print(f"{n:>6} {mode:>10} {'out of memory, stopping':>27}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break


if __name__ == "__main__":
    main()
