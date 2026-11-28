"""Download the robot models (pinned commits, into third_party/, gitignored).

    python scripts/fetch_menagerie.py            # Franka Panda (MuJoCo Menagerie, ~37 MB)
    python scripts/fetch_menagerie.py --fanuc    # FANUC CRX-10iA/L (~7 MB, needs git-lfs + xacro)
    python scripts/fetch_menagerie.py --all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paintbot.mujoco_cell import assets  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanuc", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or not args.fanuc:
        print(f"Franka Panda @ {assets.MENAGERIE_COMMIT[:7]} -> {assets.menagerie_root()}")
        print(f"OK: {assets.fetch()}")
    if args.all or args.fanuc:
        print(f"FANUC CRX-10iA/L @ {assets.FANUC_COMMIT[:7]} -> {assets.fanuc_root()}")
        print(f"OK: {assets.fetch_fanuc()}")
