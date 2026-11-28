# Running the painting cell in Isaac Sim on AWS

Isaac Sim can't run on the dev laptop (4 GB VRAM), so it runs on a cloud GPU and
you watch it from the laptop. This folder has everything for that.

| Step | Where | Needs Isaac? | Verified |
|---|---|---|---|
| `prepare_assets.py`: robot URDF, expert trajectories, panel USDs | cloud box (or laptop) | no | yes, `tests/test_isaac_task.py` |
| `run_paint.py`: Isaac Lab 3.0 env, plays the expert, paints, prints coverage | cloud box | yes | **not yet**: first run is on the cloud |

What runs in Isaac: the FANUC CRX-10iA/L on a ±0.8 m rail with the spray gun, the
same cell that passes all MuJoCo tests (`paintbot/cell_layout.py` is shared). The
joint path comes from the MuJoCo-verified planner + IK, so Isaac only does physics,
rendering and paint (BatchedPaintSim on the GPU, all envs at once). The expected
coverage is stored with each expert file and printed next to Isaac's result.

## 1. Get a GPU machine (AWS, L40S)

Isaac Sim needs an RTX-class GPU with RT cores: on AWS that's the **g6e** family
(NVIDIA L40S, 48 GB). Not A100/H100 (no RT cores).

**Recommended: NVIDIA's Isaac Automator** (the cloud path the Isaac Lab 3.0 docs
describe). It creates the instance with drivers, Isaac Sim and Isaac Lab, and gives
you a remote desktop:

```bash
git clone https://github.com/isaac-sim/IsaacAutomator.git && cd IsaacAutomator
# follow its README for AWS credentials and the instance type (pick a g6e / L40S)
./deploy-aws --isaaclab v3.0.0
./ssh <name>        # terminal on the box
./novnc <name>      # its desktop in your browser; NoMachine also works (smoother)
```

Docs: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/features/cloud_deployment.html

**Manual alternative:** a g6e EC2 instance with Ubuntu 22.04+ and NVIDIA driver
≥ 580.65.06, then Isaac Lab 3.0 (Python 3.12, uv) per
https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html , and
NoMachine for the desktop. Restrict the security group's SSH/NoMachine ports to
your own IP.

Billing is per hour: **stop the instance when you're done**. Keep its disk if you
don't want to reinstall Isaac Sim (tens of GB) next time.

## 2. Prepare the assets (on the box, once)

Use a separate small Python env, not the Isaac one (`usd-core` would clash with the
USD that Isaac Sim bundles):

```bash
sudo apt-get install -y git-lfs && git lfs install
git clone https://github.com/Abhishek-Sewar/VLA-Paint-Project.git paintbot && cd paintbot
uv venv prep --python 3.12 && source prep/bin/activate
uv pip install -e ".[isaac-prep]"
python scripts/isaac/prepare_assets.py      # fetches the CRX, writes outputs/isaac/
deactivate
```

It prints the MuJoCo reference coverage for each panel (flat 0.982, door 0.869,
curved 0.982).

## 3. Run it in Isaac (inside the remote desktop)

In the Isaac Lab environment (Automator: see its README for how to activate it):

```bash
cd paintbot && uv pip install -e .            # paintbot core into the Isaac env
python scripts/isaac/run_paint.py --help      # shows Isaac Lab's launcher flags too
python scripts/isaac/run_paint.py --panel door --num_envs 16 --viz kit
python scripts/isaac/run_paint.py --panel curved --num_envs 64 --video   # MP4 in outputs/isaac/videos
```

`--viz kit` opens the Isaac Sim window in the remote desktop; env 0's panel shows
the coat building up row by row. With `--video`, copy the MP4 to the laptop with
`scp`.

## First run: what may need fixing

`env.py` is written against the Isaac Lab 3.0 docs but has never run. The likely
spots, all isolated in small helpers at the top of `paintbot/isaac_task/env.py`:

- data access (`.torch` on ProxyArrays), joint targets and joint-state writes
  (3.0 renamed these; the code uses the names 3.0 still accepts, with a warning)
- quaternion order for the panel pose (3.0 uses x, y, z, w)
- the stage path of env 0's panel mesh (live colours turn off with a message if wrong)
- `UrdfConverterCfg` field names and `VideoRecorderCfg`'s import path

If the box is set up as an SSH host, Claude Code can run and fix these directly.
