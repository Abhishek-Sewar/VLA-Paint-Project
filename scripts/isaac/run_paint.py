"""Run the painting cell in Isaac Sim (Isaac Lab 3.0) on the cloud GPU.

    python scripts/isaac/prepare_assets.py                       # once, no Isaac needed
    python scripts/isaac/run_paint.py --panel door --num_envs 16 --viz kit
    python scripts/isaac/run_paint.py --panel curved --num_envs 64 --video  # MP4 in outputs/isaac/videos

The FANUC CRX plays the scripted expert (raster planner + IK, verified in MuJoCo)
in every env; paint comes from BatchedPaintSim on the GPU. At the end it prints
coverage per env next to the MuJoCo reference for the same episode.

Launcher flags (e.g. --viz kit / none) come from Isaac Lab's own argument parser;
run with --help on the cloud box to see the ones your Isaac Lab version has.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--panel", default="door", choices=["flat", "door", "curved"])
parser.add_argument("--robot", default="crx10ia_l")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--assets", default="outputs/isaac")
parser.add_argument("--video", action="store_true", help="record an MP4 of the run")
parser.add_argument("--episodes", type=int, default=1)

# Isaac Lab 3.0 deprecates AppLauncher in favour of launch_simulation(), but still
# ships it; it is the simplest way to start the app *before* converting the URDF.
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

# ---- everything below needs the running app --------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from paintbot.isaac_task.env import PaintEnv, make_env_cfg  # noqa: E402
from paintbot.isaac_task.expert import load_expert  # noqa: E402


def convert_robot(urdf: Path, usd_dir: Path) -> str:
    cfg = UrdfConverterCfg(
        asset_path=str(urdf), usd_dir=str(usd_dir), fix_base=True,
        merge_fixed_joints=False,            # keep the `nozzle` link
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)))
    return UrdfConverter(cfg).usd_path


def main():
    root = Path(args.assets)
    urdf = root / f"{args.robot}_paintcell.urdf"
    expert = root / f"expert_{args.robot}_{args.panel}.npz"
    panel_usd = root / f"panel_{args.panel}.usda"
    for f in (urdf, expert, panel_usd):
        if not f.is_file():
            sys.exit(f"missing {f}; run scripts/isaac/prepare_assets.py first")

    robot_usd = convert_robot(urdf, root / "usd")
    print(f"[paintbot] robot USD: {robot_usd}")
    cfg = make_env_cfg(str(expert), robot_usd, str(panel_usd.resolve()), args.num_envs)
    if args.video:
        try:
            from isaaclab.envs import VideoRecorderCfg
        except ImportError:  # location differs between 3.x releases
            from isaaclab.envs.utils import VideoRecorderCfg
        cfg.video_recorders = [VideoRecorderCfg(source="viz:kit",
                                                output_dir=str(root / "videos"),
                                                video_length=int(cfg.episode_length_s * 30))]
    env = PaintEnv(cfg)
    ref = load_expert(expert)["meta"]["mujoco_reference"]

    actions = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
    env.reset()
    for ep in range(args.episodes):
        for _ in range(env.T - 1):
            if not app.is_running():
                break
            env.step(actions)
        m = env.last_metrics or env.metrics_now()
        cov = np.asarray(m["coverage"])
        print(f"[paintbot] episode {ep}: coverage mean {cov.mean():.3f} "
              f"(min {cov.min():.3f}, max {cov.max():.3f}) over {env.num_envs} envs; "
              f"MuJoCo reference {ref['coverage']:.3f}; uniformity_cv "
              f"{np.mean(m['uniformity_cv']):.3f} vs {ref['uniformity_cv']:.3f}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
