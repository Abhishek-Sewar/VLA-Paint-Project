# paintbot: project context for Claude Code

## What this is
A rebuild of a language-conditioned robotic spray-painting system (vehicle/train
panels) with a modern stack. A robot arm takes an instruction like "apply primer to
the upper right door panel", finds the region, and paints it with an even coat while
avoiding glass and already-painted areas.

The original (2024–25) used Isaac Sim + frozen CLIP + a small MLP action head,
single-step joint-delta actions, and 6 rounds of hand-tuned domain randomization
(38% → 84% real-robot task completion). This rebuild fixes its known flaws and uses
current tools. Do not include anything from the original client project (client
name, data, code, documents) in this repo; it is public.

## Hardware and environment
- Windows 11 laptop, RTX 3050 (4–6 GB VRAM), ~44 GB free disk.
- Conda env `py310` (Python 3.10.13) with numpy, matplotlib, pytorch.
- Run commands from `C:\Users\abhis\projects\paintbot` in Anaconda Prompt (cmd, not bash).
- Isaac Sim cannot run locally (needs 16 GB VRAM, RT cores). Isaac work happens on
  rented cloud GPUs with RT cores (L40S / RTX 4090-class), never A100/H100 for Isaac.
  Policy training can use A100/H100. Don't install Isaac Sim, CUDA toolkits, or large
  datasets locally.
- Repo: https://github.com/Abhishek-Sewar/VLA-Paint-Project (branch `main`).

## Code layout
- `paintbot/panel.py`    Panel grid (u, v in metres, centre 0,0; +u right, +v up as
                          seen by a viewer at +z). Curvature = cylindrical bend about the
                          vertical axis, convex toward the robot. Cutouts = no-paint zones.
- `paintbot/spray.py`    NumPy reference deposition model: Gaussian cone, sigma =
                          distance*tan(half_angle)/2, cos-incidence factor, transfer
                          efficiency vs distance. Thickness in microns. Windowed by default.
- `paintbot/planner.py`  Raster (boustrophedon) planner = scripted expert. Pass speed is
                          derived from target thickness. `simulate()` runs a trajectory.
- `paintbot/metrics.py`  coverage, in_spec, pooling, uniformity_cv, overspray_nontarget,
                          glass_contamination, transfer_efficiency, accept (coverage ≥ 0.8).
- `paintbot/validation.py` Data-quality checks per episode (collision, joint limits,
                          black frames, depth NaN, spray miss via aim point, coverage).
- `paintbot/torch_sim.py` Batched PyTorch version (Phase 2a). Per-env curvature,
                          cutouts, target, gun params. `window_half_m=0.2` for speed,
                          `None` = exact full grid.
- `paintbot/mujoco_cell/` MuJoCo cell (Phase 2b-local). Only module that imports mujoco.
    - `scene.py`   MjSpec build: Menagerie Panda (`panda_nohand.xml`) on a ±0.8 m y-rail
                   atop a 0.3 m pedestal, gun at 90° to the flange (nozzle site +z =
                   spray axis), panel = textured mesh from `Panel.surface()` + back face +
                   convex collision slab. `PanelPose`: panel u→−y, v→+z, normal→−x,
                   centre (0.70, 0, 0.55). 30 Hz control, 20 physics substeps.
    - `ik.py`      Task-priority DLS IK: nozzle pos + spray axis (primary), "flange
                   down" roll (secondary), posture with rail under wrist (tertiary),
                   active-set joint limits. `solve_trajectory` raises on joint jumps.
    - `cell.py`    `PaintCell(panel, target, mode="dynamic"|"kinematic")`: plan → IK →
                   approach from park pose → per tick move arm, paint with
                   `BatchedPaintSim` from the *measured* nozzle pose, log for
                   `validate_episode`. Dynamic mode uses velocity feedforward.
    - `texture.py` thickness → RGB (row 0 = panel bottom), writes `model.tex_data`.
    - `assets.py`  Menagerie path (`third_party/mujoco_menagerie`, gitignored, pinned
                   commit; override with `PAINTBOT_MENAGERIE`).
- `tests/test_core.py`, `tests/test_torch_sim.py`, `tests/test_mujoco_cell.py`
                          Run as `python tests\<file>.py` or all at once with
                          `python -m pytest -q` (pytest is in py310; ~2 min total).
                          torch tests compare against NumPy; mujoco tests compare against
                          a direct BatchedPaintSim run and skip if mujoco/assets missing.
- `scripts/demo.py`, `scripts/bench_torch.py`, `scripts/mujoco_paint.py` (live viewer,
  `--headless` for metrics), `scripts/fetch_menagerie.py`

## Ground rules
- `spray.py` (NumPy) is the reference. Any change to the physics goes there first,
  then into `torch_sim.py`, and `tests/test_torch_sim.py` must still pass.
- Keep core code simulator-agnostic; Isaac-specific code goes in a separate module.
- Run the tests after every change. Commit working steps with clear messages.
- Known, intentional limitation: window edges trade coverage vs glass overspray
  (`cutout_margin_sigma`). Don't "fix" it by loosening metrics.

## Status
- Phase 1 done and verified on this laptop (tests pass; demo coverage 0.98 flat/curved,
  0.87 door with window).
- Phase 2a done and verified on this laptop with real PyTorch (2.11.0+cu128):
  `torch.cuda.is_available()` is True on the RTX 3050, all 7 torch tests pass
  including GPU-vs-CPU, no code changes were needed. Benchmark at 1024 envs:
  ~33k env-steps/s window mode, ~10k full grid (full table in README).
- Phase 2b-local (MuJoCo cell) done on this laptop with mujoco 3.17.0: all 9 cell
  tests pass. Kinematic mode matches a direct BatchedPaintSim run to ~1e-5; dynamic
  mode tracks within 6 mm max / 1.4 mm mean while spraying, same coverage, no
  collisions, validation passes on flat, door and curved. Layout lessons: a fixed
  Panda can't reach the curved door's edges (hence the rail), an in-line gun drove
  joint 6 into its limit (hence the 90° gun), and the Panda's own `home` pose puts
  the gun inside the panel (hence the park pose).
- Next: Phase 2b (Isaac Lab task wrapper, on cloud GPUs).

## Roadmap
1. Phase 1: paint physics, planner, metrics, validation (NumPy). Done.
2. Phase 2: Isaac Lab env (cloud). 2a batched torch paint sim (done, verified on GPU);
   2b-local MuJoCo cell with Panda + rail + IK + live texture (done);
   2b Isaac Lab task wrapper: 7-joint arm, 8 panel configs as USD, RGB+depth cameras,
   domain randomization (reflectivity, lighting, arm reflection, depth noise, gun params).
3. Phase 3: demo generation at scale: planner + cuRobo IK in parallel envs, validation
   on every episode, LeRobot dataset format, LLM-paraphrased instructions. Hold out
   configs 7–8 entirely.
4. Phase 4: photoreal augmentation with Cosmos Transfer on a subset.
5. Phase 5: policies. Baseline ACT/Diffusion Policy (LeRobot), then π0.5 LoRA via
   openpi with action chunks + spray on/intensity as action dims. In parallel, hybrid
   track: VLM + text-prompted segmentation picks region → planner paints it. Compare.
6. Phase 6: closed-loop eval harness: held-out configs, failure triggers (timeout,
   velocity freeze, low coverage at midpoint, out of bounds), failure classification.

## Fixes to carry over from the original design
- Feed patch tokens (not a pooled vector) into any cross-attention.
- Mirroring augmentation must also mirror actions and swap left/right in the text.
- Use depth as a model input, and predict spray state as part of the action.
- Spray-miss check uses where the spray axis hits the panel, not end-effector position.
