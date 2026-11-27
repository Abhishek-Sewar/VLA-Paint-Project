# paintbot

Phase 1 of the VLA spray-painting rebuild: the simulator-agnostic core. It runs on any
laptop (CPU only, no Isaac Sim needed) and everything later builds on it.

| Module | What it does |
|---|---|
| `paintbot/panel.py` | Panel surface grid (flat or curved), window cutouts, region masks |
| `paintbot/spray.py` | Gaussian-cone paint deposition: distance, tilt and transfer efficiency |
| `paintbot/planner.py` | Raster coverage planner, the scripted expert that generates demos |
| `paintbot/metrics.py` | Coverage, in-spec, pooling, uniformity, overspray, glass contamination |
| `paintbot/validation.py` | The original 4 data-quality checks, ported and fixed |

| `paintbot/torch_sim.py` | Batched PyTorch version of the paint model (Phase 2a), CPU or GPU |

## Setup

### Conda (recommended; what this was developed on)

Windows 11, Anaconda Prompt, conda env `py310` (Python 3.10):

```bat
conda create -n py310 python=3.10
conda activate py310

:: PyTorch with CUDA (pick the CUDA build for your driver at https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu128

cd paintbot
pip install -e ".[dev]"

:: Check everything works
python -c "import torch; print(torch.cuda.is_available())"
python tests\test_core.py
python tests\test_torch_sim.py
:: Prints metrics, writes outputs\coverage_demo.png
python scripts\demo.py
```

Install PyTorch from the PyTorch index as above: a plain `pip install torch` on Windows
gives a CPU-only build. If `torch.cuda.is_available()` prints `False`, that's why.
Everything still runs on CPU, only slower.

### uv (alternative)

```bash
# Install uv
#    Windows PowerShell:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    Linux / macOS:       curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 3.11 matches Isaac Sim 5.x, so code ports cleanly later
cd paintbot
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build; skip for CPU-only
uv pip install -e ".[dev,torch]"

uv run pytest -q
uv run python scripts/demo.py
```

## Expected demo output (abridged)

```
Flat panel               coverage 0.982  uniformity_cv 0.085  ~330x real time
Door with window         coverage 0.869  glass_contamination 0.284
Flat panel, upper right  coverage 0.964  overspray_nontarget 0.080
Curved door              coverage 0.982  uniformity_cv 0.085
```

## Known limitations (on purpose, for now)

- **Window edges** are a real trade-off: a 15-degree gun can't hit the frame around
  glass without some overspray. `cutout_margin_sigma` trades coverage for
  contamination. Real shops fix this with masking or a narrow detail gun, and it's a
  good place for the learned policy to beat the scripted planner later.
- `region_mask()` understands only simple words (upper/lower/left/right/full). A
  later phase replaces it with a vision-language model plus segmentation.
- No occlusion or paint flow (runs and sags). Pooling is detected by thickness only.

## Phase 2a: batched PyTorch simulator

`paintbot/torch_sim.py` runs many environments at once on CPU or GPU, each with its
own panel shape, cutouts, target region and randomized gun parameters. It is checked
against the NumPy model by `tests/test_torch_sim.py`.

```bash
python tests/test_torch_sim.py     # should print 7 PASS lines
python scripts/bench_torch.py      # env-steps per second on your GPU
```

Use `window_half_m=0.2` for speed: each step only touches the patch around where the
spray lands. `window_half_m=None` touches every cell and is the exact reference.

### Benchmark

`scripts/bench_torch.py` on an RTX 3050 Laptop GPU (4 GB), PyTorch 2.11 + CUDA 12.8,
float32, 1.0 x 0.6 m panel at 5 mm cells, half flat and half curved panels. "x real
time" is env-steps/s divided by the 30 Hz control rate.

| envs | window: env-steps/s | window: x real time | full grid: env-steps/s | full grid: x real time |
|---:|---:|---:|---:|---:|
| 1 | 298 | 10 | 634 | 21 |
| 16 | 4,755 | 158 | 8,212 | 274 |
| 64 | 15,911 | 530 | 9,805 | 327 |
| 256 | 20,405 | 680 | 10,263 | 342 |
| 1024 | 32,876 | 1,096 | 10,372 | 346 |

Window mode keeps scaling with env count; full grid levels off around 10k env-steps/s
from 64 envs on. Below 64 envs window mode is slower than full grid, likely because
its extra per-step kernel launches dominate when there is little work per launch.

## Next step

Wrap `BatchedPaintSim` in an Isaac Lab environment so the robot arm drives the gun.
