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
| `paintbot/cell_layout.py` | Cell layout shared by MuJoCo and Isaac: robot mount, rail, panel pose, gun geometry |
| `paintbot/mujoco_cell/` | MuJoCo cell: Franka Panda or FANUC CRX paints the panel, coat shown live (Phase 2b-local) |
| `paintbot/isaac_task/` | Isaac Lab 3.0 task for the cloud GPU: robot URDF, expert playback, panel USD (Phase 2b) |

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

## Phase 2b-local: MuJoCo painting cell

A robot arm with a spray gun follows the `RasterPlanner` trajectory using inverse
kinematics. `BatchedPaintSim` paints from the gun's *measured* pose, and the film
shows live as a texture on the panel. Two robots:

- **FANUC CRX-10iA/L** (`--robot crx10ia_l`): industrial 6-axis arm, 1.42 m reach;
  FANUC sells a paint version of it. Model from FANUC's official
  [fanuc_description](https://github.com/FANUC-CORPORATION/fanuc_description) (Apache-2.0).
- **Franka Panda** (default): 7-axis research arm from
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

```bat
pip install -e ".[mujoco]"
:: Robot models, pinned commits, into third_party\ (gitignored). FANUC needs git-lfs.
python scripts\fetch_menagerie.py --all

python scripts\mujoco_paint.py --robot crx10ia_l --panel curved
python scripts\mujoco_paint.py --panel curved
python scripts\mujoco_paint.py --panel door --region "upper right" --speed 2
:: No window, just metrics + validation
python scripts\mujoco_paint.py --panel door --headless
python tests\test_mujoco_cell.py
```

Two modes: `--mode dynamic` (default) drives the Panda's position actuators, so
tracking lag shows up in the paint; `--mode kinematic` sets joint angles directly
and matches a robot-free `BatchedPaintSim` run to ~1e-5.

**Cell layout**, and why:

- **Both robots on a ±0.8 m linear rail** (an extra axis, as in real paint cells).
  A fixed-base Panda reaches only the middle 60% of the 1.0 m curved door: near its
  edges the surface normal turns ~45° and pushes the wrist ~0.8 m sideways. Even
  the 1.42 m CRX can't reach that door's top corners from a fixed base with the gun
  hanging under the flange.
- **Panda's rail on a 0.3 m pedestal** (the CRX's on a 6 cm track). The gun hangs
  under the wrist, so from floor level the Panda couldn't reach the top corners.
- **Gun at 90° to the flange**, like a real spray gun. In line with the flange,
  spraying horizontally drove the Panda's joint 6 into its limit.
- **IK** (`ik.py`) keeps nozzle position and spray axis exact, prefers "flange
  down" (so yawing the spray is the last wrist joint), then a comfortable posture with the rail
  under the wrist. It raises if the joint path would jump.

Measured (dynamic mode, spray on):

| Robot | Panel | coverage (cell / no robot) | max / mean tracking error | validation |
|---|---|---|---|---|
| CRX-10iA/L | Flat | 0.982 / 0.982 | 5.5 mm / 0.9 mm, 0.1° | pass |
| CRX-10iA/L | Door with window | 0.869 / 0.869 | 5.5 mm / 1.0 mm, 0.1° | pass |
| CRX-10iA/L | Curved door | 0.982 / 0.982 | 8.6 mm / 1.6 mm, 0.5° | pass |
| Panda | Flat | 0.982 / 0.982 | 4.7 mm / 0.9 mm, 0.2° | pass |
| Panda | Door with window | 0.870 / 0.869 | 4.7 mm / 0.9 mm, 0.2° | pass |
| Panda | Curved door | 0.982 / 0.982 | 6.0 mm / 1.4 mm, 0.5° | pass |

The cell reproduces the planner's limits too: door + `"upper right"` fails
`low_coverage` (0.743, vs 0.744 with no robot) because that small region borders
the window, where the planner switches the spray off (see Known limitations).

The viewer keeps up with real time on the RTX 3050 laptop and tops out around
2.5x (`--speed`); `--headless` runs a 50 s episode in about 6 s.

## Phase 2b: Isaac Sim on a cloud GPU

The same CRX cell in Isaac Lab 3.0, on an AWS g6e (L40S) instance, watched from the
laptop through a remote desktop. `paintbot/isaac_task/` builds one URDF for rail +
CRX + gun + nozzle (checked against the MuJoCo cell), plays back the MuJoCo-verified
expert joint path, and paints every env with `BatchedPaintSim` on the GPU, showing
env 0's coat live as panel vertex colours. The inputs are tested locally
(`tests/test_isaac_task.py`); the Isaac env itself hasn't run yet.

Setup and run: [scripts/isaac/README.md](scripts/isaac/README.md).

## Next steps

Run the Isaac task on AWS and fix what Isaac Lab 3.0 needs, then cameras (RGB + depth),
8 panel configurations and domain randomization. Policies (Phase 5): ACT/Diffusion
baseline, then NVIDIA GR00T N1.6 fine-tuned on the generated demos.
