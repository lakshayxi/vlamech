"""What exactly produced a result: code, upstream code, packages, model files and scientific assets.

Identity is verified, never trusted from version strings:
  * our code        - git commit + dirty flag of the project (results/ excluded)
  * upstream code   - each external/ checkout's HEAD and dirty flag, compared with the pins in env/setup_env.sh
  * packages        - pip direct_url commit of the transformers fork and dlimp (a version number cannot identify a fork)
  * model           - SHA-256 of every model file compared with configs/models.lock.yaml (once per process)
  * assets          - SHA-256 of fixture layouts and calibration caches
Any mismatch raises before a model is loaded.
"""
from __future__ import annotations

import functools
import re
import subprocess
from importlib import metadata
from pathlib import Path

from vlamech.utils.io import PROJECT_ROOT, file_sha256, from_json, git_state, load_yaml

EXTERNAL = {"openvla-oft": "OPENVLA_OFT_COMMIT", "LIBERO": "LIBERO_COMMIT"}
PACKAGES = {"transformers": "TRANSFORMERS_FORK_COMMIT", "dlimp": "DLIMP_COMMIT"}
VERSIONS = ("torch", "numpy", "mujoco", "robosuite", "tensorflow-cpu", "opencv-python", "protobuf")
# Per policy: (setup script declaring the package pins, VCS-installed packages, versions recorded). pi0.5 runs in its
# own environment (env/setup_env_lerobot.sh): LeRobot's transformers, no TensorFlow / OFT data stack.
POLICY_ENVS = {
    "oft": ("setup_env.sh", PACKAGES, VERSIONS),
    "pi05": ("setup_env_lerobot.sh", {"transformers": "PI05_TRANSFORMERS_COMMIT", "lerobot": "LEROBOT_COMMIT"},
             ("torch", "numpy", "mujoco", "robosuite", "opencv-python-headless")),
}


def pins(policy: str = "oft") -> dict:
    """Commit pins declared in env/setup_env.sh (external checkouts) and the policy's setup script (packages)."""
    script, packages, _ = POLICY_ENVS[policy]
    text = (PROJECT_ROOT / "env" / "setup_env.sh").read_text()
    if script != "setup_env.sh":
        text += "\n" + (PROJECT_ROOT / "env" / script).read_text()
    out = dict(re.findall(r"^([A-Z0-9_]+_COMMIT)=([0-9a-f]{40})", text, flags=re.M))
    missing = (set(EXTERNAL.values()) | set(packages.values())) - set(out)
    if missing:
        raise RuntimeError(f"pins missing from env/setup_env.sh: {sorted(missing)}")
    return out


def _git(repo: Path, *args) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()


def upstream_identity(strict: bool = True, policy: str = "oft") -> dict:
    p = pins(policy)
    _, packages, versions = POLICY_ENVS[policy]
    out = {}
    for name, var in EXTERNAL.items():
        repo = PROJECT_ROOT / "external" / name
        head = _git(repo, "rev-parse", "HEAD")
        dirty = bool(_git(repo, "status", "--porcelain"))
        out[name] = {"commit": head, "dirty": dirty, "pinned": p[var]}
        if strict and (head != p[var] or dirty):
            raise RuntimeError(f"external/{name} is at {head} (dirty={dirty}), pinned {p[var]}: rebuild with env/setup_env.sh")
    for pkg, var in packages.items():
        try:
            info = from_json(metadata.distribution(pkg).read_text("direct_url.json") or "{}")
        except metadata.PackageNotFoundError:
            info = {}
        commit = info.get("vcs_info", {}).get("commit_id")
        out[f"pkg:{pkg}"] = {"commit": commit, "pinned": p[var], "version": metadata.version(pkg) if commit else None}
        if strict and commit != p[var]:
            raise RuntimeError(f"installed {pkg} comes from {commit}, pinned {p[var]}")
    for pkg in versions:
        out[f"pkg:{pkg}"] = {"version": metadata.version(pkg)}
    return out


def code_identity(strict: bool = True, policy: str = "oft") -> dict:
    g = git_state()
    return {"project": {"commit": g["commit"], "dirty": g["dirty"]}, "upstream": upstream_identity(strict, policy)}


@functools.lru_cache(maxsize=None)
def verify_model_files(repo_id: str) -> dict:
    """Hash every pinned file of a model snapshot and compare with configs/models.lock.yaml (once per process)."""
    from huggingface_hub import snapshot_download

    from vlamech.models import pi05

    if repo_id == pi05.REPO_ID:  # never import the OFT module in the LeRobot environment (prismatic, TF)
        allow, ignore = pi05.DOWNLOAD_ALLOW, None
    else:
        from vlamech.models.oft import DOWNLOAD_ALLOW as allow, DOWNLOAD_IGNORE as ignore

    lock = load_yaml(PROJECT_ROOT / "configs" / "models.lock.yaml")[repo_id]
    path = Path(snapshot_download(repo_id, revision=lock["revision"], allow_patterns=allow,
                                  ignore_patterns=ignore, local_files_only=True))
    bad = {f: h for f, h in lock["files"].items() if file_sha256(path / f) != h}
    if bad:
        raise RuntimeError(f"{repo_id}: files differ from configs/models.lock.yaml: {sorted(bad)}")
    return {"repo": repo_id, "revision": lock["revision"], "n_files": len(lock["files"]),
            "lock_sha256": file_sha256(PROJECT_ROOT / "configs" / "models.lock.yaml")}


def files_sha256(paths) -> dict:
    """{path relative to the project: sha256} for existing files (sorted)."""
    out = {}
    for p in sorted({Path(p).resolve() for p in paths}):
        if not p.exists():
            raise FileNotFoundError(p)
        try:
            key = str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            key = str(p)
        out[key] = file_sha256(p)
    return out
