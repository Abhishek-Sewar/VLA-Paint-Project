"""Download the Franka Panda model (MuJoCo Menagerie, ~37 MB) at a pinned commit.

    python scripts/fetch_menagerie.py

Goes into third_party/mujoco_menagerie/ (gitignored), or $PAINTBOT_MENAGERIE.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paintbot.mujoco_cell import assets  # noqa: E402

if __name__ == "__main__":
    print(f"Fetching {assets.ROBOT_DIR} @ {assets.MENAGERIE_COMMIT[:7]} "
          f"into {assets.menagerie_root()} ...")
    print(f"OK: {assets.fetch()}")
