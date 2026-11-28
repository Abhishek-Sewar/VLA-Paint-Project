"""Robot models: where they live and how to fetch them. None are committed.

* Franka Panda, MuJoCo Menagerie (Apache-2.0, ~37 MB) -> third_party/mujoco_menagerie/
* FANUC CRX-10iA/L, FANUC's official fanuc_description (Apache-2.0, ~7 MB for this
  model) -> third_party/fanuc_description/. Meshes are stored with Git LFS, so
  fetching needs git-lfs. The xacro is expanded to a plain URDF (needs the `xacro`
  pip package) with absolute mesh paths, cached under build/.

Both are sparse, shallow clones at pinned commits. Override the locations with
PAINTBOT_MENAGERIE / PAINTBOT_FANUC.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "eea4c7431b63da8278b0ff31638e0b0e941cec0e"
ROBOT_DIR = "franka_emika_panda"
ROBOT_XML = "panda_nohand.xml"
DEFAULT_ROOT = _REPO_ROOT / "third_party" / "mujoco_menagerie"

FANUC_URL = "https://github.com/FANUC-CORPORATION/fanuc_description.git"
FANUC_COMMIT = "fb40c9803a826ba68c7c8e28ba904a25efa7fcd2"     # release v2.3.0
FANUC_ROOT = _REPO_ROOT / "third_party" / "fanuc_description"
FANUC_PKG = "fanuc_crx_description"


# ----------------------------------------------------------------- helpers
def _sparse_fetch(root: Path, url: str, commit: str, patterns: list[str], lfs: bool = False):
    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True)

    if not (root / ".git").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        git("init", "-q")
        git("remote", "add", "origin", url)
    git("sparse-checkout", "set", "--no-cone", *patterns)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "-q", "--detach", "FETCH_HEAD")
    if lfs:
        inc = ",".join(p.lstrip("/") + ("**" if p.endswith("/") else "") for p in patterns)
        git("lfs", "pull", "--include", inc)


# ------------------------------------------------------------------- Panda
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


def available(robot: str = "panda") -> bool:
    try:
        panda_xml() if robot == "panda" else fanuc_urdf(robot)
        return True
    except (FileNotFoundError, ImportError):
        return False


def fetch(root: Path | None = None) -> Path:
    """Sparse-clone franka_emika_panda/ at MENAGERIE_COMMIT. Safe to re-run."""
    root = Path(root or menagerie_root())
    _sparse_fetch(root, MENAGERIE_URL, MENAGERIE_COMMIT, [f"/{ROBOT_DIR}/"])
    return root / ROBOT_DIR / ROBOT_XML


# ------------------------------------------------------------------- FANUC
def fanuc_root() -> Path:
    return Path(os.environ.get("PAINTBOT_FANUC", FANUC_ROOT))


def fetch_fanuc(model: str = "crx10ia_l", root: Path | None = None) -> Path:
    """Sparse-clone one CRX model's macro + meshes at FANUC_COMMIT (needs git-lfs)."""
    root = Path(root or fanuc_root())
    _sparse_fetch(root, FANUC_URL, FANUC_COMMIT, [
        "/LICENSES/", "/README.md", f"/{FANUC_PKG}/package.xml",
        f"/{FANUC_PKG}/urdf/{model}_urdf_macro.xacro",
        f"/{FANUC_PKG}/meshes/{model}/"], lfs=True)
    return fanuc_urdf(model, root)


def fanuc_urdf(model: str = "crx10ia_l", root: Path | None = None) -> Path:
    """Plain URDF for a FANUC model (base_link at the origin, `fanuc_flange` link at the
    tool flange, absolute mesh paths). Built from the xacro on first use."""
    root = Path(root or fanuc_root())
    pkg = root / FANUC_PKG
    macro = pkg / "urdf" / f"{model}_urdf_macro.xacro"
    if not macro.is_file():
        raise FileNotFoundError(
            f"FANUC {model} model not found at {macro}.\n"
            "Fetch it with:  python scripts\\fetch_menagerie.py --fanuc")
    stl = next((pkg / "meshes" / model / "collision").glob("*.stl"), None)
    if stl is None or stl.stat().st_size < 1024:          # LFS pointer, not a mesh
        raise FileNotFoundError(f"{model} meshes are Git LFS pointers; install git-lfs "
                                "and re-run scripts\\fetch_menagerie.py --fanuc")
    out = root / "build" / f"{model}.urdf"
    if out.is_file() and out.stat().st_mtime >= macro.stat().st_mtime:
        return out

    import xacro                                           # pip install xacro
    wrapper = (f'<?xml version="1.0"?><robot name="{model}" '
               'xmlns:xacro="http://wiki.ros.org/xacro">'
               f'<xacro:include filename="{macro.as_posix()}"/>'
               '<link name="world"/><link name="tool0"/>'
               f'<xacro:{model} parent="world" child="tool0">'
               f'<origin xyz="0 0 0" rpy="0 0 0"/></xacro:{model}></robot>')
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path = out.with_suffix(".xacro")
    wrapper_path.write_text(wrapper)
    urdf = xacro.process_file(str(wrapper_path)).toxml()
    urdf = re.sub(r"package://([^/]+)/", lambda m: (root / m.group(1)).as_posix() + "/", urdf)
    out.write_text(urdf)
    return out
