"""Where the Franka Panda model from MuJoCo Menagerie lives, and how to fetch it.

The meshes are ~37 MB (Apache-2.0), so they are not committed. `fetch()` does a
sparse, shallow clone of just `franka_emika_panda/` at a pinned commit into
`third_party/mujoco_menagerie/` (gitignored). Set PAINTBOT_MENAGERIE to use a
Menagerie checkout somewhere else.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "eea4c7431b63da8278b0ff31638e0b0e941cec0e"
ROBOT_DIR = "franka_emika_panda"
ROBOT_XML = "panda_nohand.xml"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _REPO_ROOT / "third_party" / "mujoco_menagerie"


def menagerie_root() -> Path:
    return Path(os.environ.get("PAINTBOT_MENAGERIE", DEFAULT_ROOT))


def panda_xml() -> Path:
    """Path to panda_nohand.xml. Raises FileNotFoundError with the fix if missing."""
    p = menagerie_root() / ROBOT_DIR / ROBOT_XML
    if not p.is_file():
        raise FileNotFoundError(
            f"Franka Panda model not found at {p}.\n"
            "Fetch it with:  python scripts\\fetch_menagerie.py")
    return p


def available() -> bool:
    try:
        panda_xml()
        return True
    except FileNotFoundError:
        return False


def fetch(root: Path | None = None) -> Path:
    """Sparse-clone franka_emika_panda/ at MENAGERIE_COMMIT. Safe to re-run."""
    root = Path(root or menagerie_root())

    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True)

    if not (root / ".git").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        git("init", "-q")
        git("remote", "add", "origin", MENAGERIE_URL)
    git("sparse-checkout", "set", "--no-cone", f"/{ROBOT_DIR}/")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT)
    git("checkout", "-q", "--detach", "FETCH_HEAD")
    return root / ROBOT_DIR / ROBOT_XML
