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

## Setup (Windows 11 or Ubuntu)

```bash
# 1. Install uv (fast Python package manager)
#    Windows PowerShell:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    Linux / macOS:       curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment (Python 3.11 matches Isaac Sim 5.x, so code ports cleanly later)
cd paintbot
uv venv --python 3.11
uv pip install -e ".[dev]"

# 3. Check everything works
uv run pytest -q
uv run python scripts/demo.py     # prints metrics, writes outputs/coverage_demo.png
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

## Next step

Phase 2 ports `spray.deposit()` to batched PyTorch so thousands of environments
paint in parallel on the GPU inside Isaac Lab.
