"""Study parameters (configs/protocol.yaml) and the selections frozen before the held-out runs.

Experiment code takes every parameter from here. `protocol()` checks on load that each parameter has a `value`, a
`source` and a non-empty `why`, so an undocumented parameter cannot be added. `frozen()` reads
configs/frozen_levels.yaml and raises if calibration has not been frozen yet.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
FROZEN_LEVELS = CONFIG_DIR / "frozen_levels.yaml"
FROZEN_MITIGATION = CONFIG_DIR / "frozen_mitigation.yaml"
SOURCES = {"upstream", "definitional", "rule", "convention"}


def _is_param(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node


def _validate(node: Any, path: str) -> None:
    if _is_param(node):
        missing = {"value", "source", "why"} - set(node)
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        if node["source"] not in SOURCES:
            raise ValueError(f"{path}: source {node['source']!r} not in {SOURCES}")
        if not str(node["why"]).strip():
            raise ValueError(f"{path}: empty justification")
        return
    if not isinstance(node, dict):
        raise ValueError(f"{path}: expected a parameter mapping with value/source/why")
    for k, v in node.items():
        _validate(v, f"{path}.{k}" if path else k)


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def protocol_tree() -> dict:
    tree = _load("protocol.yaml")
    _validate(tree, "")
    return tree


def protocol(dotted: str) -> Any:
    """Value of a protocol parameter, e.g. protocol('env.max_steps')."""
    node: Any = protocol_tree()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"protocol parameter {dotted!r} not defined in configs/protocol.yaml")
        node = node[part]
    if not _is_param(node):
        raise KeyError(f"{dotted!r} is a section, not a parameter")
    return node["value"]


def frozen(dotted: str | None = None) -> Any:
    """The frozen selection (configs/frozen_levels.yaml), or one entry of it."""
    if not FROZEN_LEVELS.exists():
        raise FileNotFoundError(f"{FROZEN_LEVELS} missing: run the severity calibration and scripts/freeze_levels.py first")
    node: Any = yaml.safe_load(FROZEN_LEVELS.read_text())
    for part in (dotted.split(".") if dotted else []):
        node = node[part]
    return node


def split_states(name: str) -> list[int]:
    lo, hi = protocol(f"splits.{name}")
    return list(range(lo, hi + 1))


def model_revision(repo_id: str) -> str:
    """Pinned commit of a model repo (configs/models.lock.yaml); models never load from an unpinned 'main'."""
    lock = _load("models.lock.yaml") or {}
    if repo_id not in lock:
        raise KeyError(f"{repo_id} not pinned in configs/models.lock.yaml: run scripts/download_models.py")
    return lock[repo_id]["revision"]
