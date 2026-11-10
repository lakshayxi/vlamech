"""Shared loading for the failure-monitor and mitigation analyses (protocol mitigation.*).

Used by: scripts/analysis/mitigation.py, scripts/analysis/failure_monitor.py, scripts/run/monitor_and_mitigation.sh.

Condition labels: "<family>[-<value>][@<camera>]", e.g. "identity", "blur-8", "target_occlusion@wrist".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vlamech.analysis.ingest import grid, load_run
from vlamech.perturbations.image import CorruptionSpec
from vlamech.utils.io import read_json


def spec_of(label: str) -> CorruptionSpec:
    name, _, cam = label.partition("@")
    fam, sep, val = name.rpartition("-") if "-" in name else ("", "", "")
    family, value = (fam, float(val)) if sep else (name, None)
    return CorruptionSpec(family, value, cameras=(cam,)) if cam else CorruptionSpec(family, value)


def arm_tag(arm: str) -> str:
    return arm.replace(":", "-")


def run_dir(root: Path, experiment: str, label: str, arm: str | None) -> Path:
    middle = "" if arm is None else f"__{arm_tag(arm)}"
    return root / "results" / experiment / f"{spec_of(label).tag}__ins-own{middle}__rep0"


def thresholds(root: Path) -> dict:
    return read_json(root / "results" / "oracle_calibration" / "thresholds.json")


def load_outcomes(root: Path, experiment: str, label: str, arm: str | None, states, thr: dict):
    """(RunData, one row per episode: task, init_state, success, collision, off_table, steps), complete runs only."""
    rd = load_run(run_dir(root, experiment, label, arm), grid(range(10), states)).require_complete()
    rows = []
    for (t, s), r in sorted(rd.records.items()):
        rows.append({"task": t, "init_state": s, "success": bool(r["success"]), "steps": int(r["steps"]),
                     "collision": bool(r["max_contact_force"]["environment"] > thr["force"][str(t)]["environment"]),
                     "off_table": bool(r["off_table"])})
    return rd, pd.DataFrame(rows)


def commanded(rd, key: tuple) -> tuple:
    """(actions_norm, query_steps, per-query instruction effect) of one episode of a cag:1 run."""
    z = np.load(rd.run_dir / "actions" / f"task{key[0]}-init{key[1]}.npz", allow_pickle=False)
    ie = np.array([x["instruction_effect"] for x in rd.records[key]["action_info"]], np.float64)
    return z["actions_norm"], z["query_steps"], ie
