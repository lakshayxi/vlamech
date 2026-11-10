"""Stimulus assets measured once from clean calibration frames (shared by runs and figures).

bg_side_lengths(): protocol corruption.bg_patch_size: bounding-box sides (px) of the movable objects in the clean
third-person frame at reset of every calibration init state, all tasks. Cached in
results/calibration_assets/bg_patch_sides.json together with EVERYTHING it depends on (rule, suite, split, camera
resolution, fixture-layout file hash, settle steps). A cache whose metadata does not match the current setup is
refused: delete it deliberately to rebuild; it is never silently reused or overwritten.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from vlamech.config import protocol, split_states
from vlamech.utils.io import PROJECT_ROOT, atomic_write_json, file_sha256, read_json

BG_SIDES_PATH = PROJECT_ROOT / "results" / "calibration_assets" / "bg_patch_sides.json"
RULE_VERSION = 1  # bump if the measurement code changes


def bg_sides_metadata() -> dict:
    suite = protocol("env.suite")
    return {"rule": "corruption.bg_patch_size", "rule_version": RULE_VERSION, "suite": suite,
            "states": split_states("calibration"), "resolution": int(protocol("env.resolution")),
            "settle_steps": int(protocol("env.settle_steps")),
            "fixture_layouts_sha256": file_sha256(PROJECT_ROOT / "assets" / f"fixture_layouts_{suite}.npz")}


def bg_side_lengths(make_env: Callable[[int], object], path: Path = BG_SIDES_PATH) -> list[int]:
    """make_env(task_id) -> LiberoTaskEnv with segmentation=True (benchmark scene)."""
    meta = bg_sides_metadata()
    if path.exists():
        cached = read_json(path)
        stale = {k: (cached.get(k), v) for k, v in meta.items() if cached.get(k) != v}
        if stale:
            raise RuntimeError(f"{path} was computed under different conditions {stale}; delete it to rebuild")
        return cached["sides"]
    sides = []
    for task_id in range(10):
        env = make_env(task_id)
        ids = env.instance_ids()
        for s in meta["states"]:
            seg = env.reset(s).agentview_seg
            for obj in env.movable_objects:
                ys, xs = np.nonzero(seg == ids[obj])
                if len(ys):
                    sides += [int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)]
    if not sides:
        raise RuntimeError("no movable object visible in calibration frames")
    atomic_write_json(path, meta | {"sides": sides})
    return sides
