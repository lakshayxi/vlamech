"""Versioned closed-loop outcome classifier (protocol oracle.outcome_classifier).

Used by: scripts/calibration/oracle_thresholds.py and the closed-loop analyses. Input: an oracle trace (vlamech.envs.trace), the INTENDED
task (the prompted one; for wrong-instruction probes not the loaded one) and thresholds frozen by the protocol
rules (oracle.displacement_threshold_rule, oracle.joint_threshold_rule). Nothing here reads the simulator.

Events (first trace row r >= 1 at which they hold; row 0 is the post-settle frame, so a predicate or
displacement that already holds there is not an event). An object counts as displaced only beyond
max(disp_m, its passive-drift floor) and a joint beyond max(joint threshold, its floor) (v2: the floors are the
motion with no robot action at all, scripts/calibration/passive_drift.py; v1 had no floors and counted a target's own
post-settle drift as progress when the task threshold was at that scale):
  goal_onset[t]     goal of task t false at row r-1 and true at row r
  target_progress   earliest of: a manipulated object displaced > disp_m from row 0; a goal joint moved
                    > joint[j] from row 0; robot contact force > 0 with a touch entity
  off_target        earliest of: a non-target object displaced > disp_m; a non-target joint moved > joint[j]
Outcome, first matching rule wins (CLASSIFIER_VERSION 2; precedence unchanged from v1):
  1 success          the intended goal holds at the last row
  2 intent_loss      wrong_goal:       some other task's goal had an onset
                     off_target_first: off_target happened, and strictly before target_progress (or without it)
  3 execution_loss   target_progress happened (subtype: what came first: joint / displaced / touched)
  4 no_progress      none of the above
Flags (do not change the outcome): incidental_wrong_goal (other goal onsets in a successful episode),
off_target_after_progress (execution_loss with later off-target activity), intended_goal_initially_true.
Ties (off_target and target_progress in the same row) are not "before": they classify as execution_loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

CLASSIFIER_VERSION = 2
OUTCOMES = ("success", "intent_loss", "execution_loss", "no_progress")


@dataclass(frozen=True)
class Thresholds:
    disp_m: float  # object displacement threshold of the intended task (m)
    joint: dict  # {joint name: threshold} (drawers m, knob rad)
    obj_floor: dict  # {object: passive-drift floor (m)}
    joint_floor: dict  # {joint: passive-drift floor}

    def obj(self, name: str) -> float:
        return max(self.disp_m, self.obj_floor[name])

    def jnt(self, name: str) -> float:
        return max(self.joint[name], self.joint_floor[name])

    def __post_init__(self):
        vals = [self.disp_m, *self.joint.values(), *self.obj_floor.values(), *self.joint_floor.values()]
        if not all(np.isfinite(v) and v >= 0 for v in vals):
            raise ValueError(f"thresholds must be finite and >= 0: {self}")


def _first(mask: np.ndarray) -> Optional[int]:
    """First row >= 1 where mask holds (mask indexed by row)."""
    idx = np.flatnonzero(mask[1:])
    return int(idx[0]) + 1 if len(idx) else None


def _min(*rows) -> Optional[int]:
    rows = [r for r in rows if r is not None]
    return min(rows) if rows else None


def events(trace: dict, task_id: int, thr: Thresholds) -> dict:
    tg = trace["targets"][task_id]
    obj, joints, ents = trace["obj_names"], trace["joint_names"], trace["entity_names"]
    missing = [j for j in joints if j not in thr.joint or j not in thr.joint_floor]
    missing += [o for o in obj if o not in thr.obj_floor]
    if missing:
        raise ValueError(f"no threshold / floor for {missing}")
    disp = np.linalg.norm(trace["obj_pos"] - trace["obj_pos"][0], axis=-1)  # (R, n_obj)
    jd = np.abs(trace["joint_q"] - trace["joint_q"][0])  # (R, n_joint)

    def moved(names):
        return {n: _first(disp[:, obj.index(n)] > thr.obj(n)) for n in names}

    def turned(names):
        return {n: _first(jd[:, joints.index(n)] > thr.jnt(n)) for n in names}

    touched = {n: _first(trace["contact"][:, ents.index(n)] > 0.0) for n in tg["touch_entities"]}
    progress = {"displaced": moved(tg["manipulated_objects"]), "joint": turned(tg["goal_joints"]), "touched": touched}
    off = {"displaced": moved(tg["non_target_objects"]), "joint": turned(tg["non_target_joints"])}
    g = trace["goals"]
    onsets = {}
    for k, t in enumerate(trace["goal_task_ids"]):
        r = _first(g[:, k] & ~np.concatenate([[True], g[:-1, k]]))
        if r is not None:
            onsets[t] = r
    # size of each off-target movement at its event row (for validating thresholds against videos)
    size = {n: float(disp[r, obj.index(n)]) for n, r in off["displaced"].items() if r is not None}
    size |= {n: float(jd[r, joints.index(n)]) for n, r in off["joint"].items() if r is not None}
    return {"goal_onset": onsets, "progress": progress, "off_target": off, "off_target_size": size,
            "target_progress": _min(*(r for d in progress.values() for r in d.values())),
            "off_target_first": _min(*(r for d in off.values() for r in d.values())),
            "intended_initially_true": bool(g[0, trace["goal_task_ids"].index(task_id)]),
            "intended_final": bool(g[-1, trace["goal_task_ids"].index(task_id)])}


def classify(trace: dict, task_id: int, thr: Thresholds) -> dict:
    ev = events(trace, task_id, thr)
    other = {t: r for t, r in ev["goal_onset"].items() if t != task_id}
    prog, off = ev["target_progress"], ev["off_target_first"]
    flags = {"intended_goal_initially_true": ev["intended_initially_true"]}
    if ev["intended_final"]:
        outcome, sub = "success", None
        flags["incidental_wrong_goal"] = sorted(other)
    elif other:
        outcome, sub = "intent_loss", "wrong_goal"
    elif off is not None and (prog is None or off < prog):
        outcome, sub = "intent_loss", "off_target_first"
    elif prog is not None:
        outcome = "execution_loss"
        sub = min((r, kind) for kind, d in ev["progress"].items() for r in d.values() if r is not None)[1]
        flags["off_target_after_progress"] = off is not None
    else:
        outcome, sub = "no_progress", None
    return {"outcome": outcome, "subtype": sub, "flags": flags, "events": ev, "classifier_version": CLASSIFIER_VERSION}
