"""Per-control-step oracle traces: everything the outcome classifier needs, from simulator
state only, so outcome thresholds can be calibrated and applied after collection.

Used by: vlamech.envs.episode.run_episode (records), vlamech.metrics.outcome (classifies),
scripts/calibration/oracle_thresholds.py (thresholds).

Row r of every array = the state after r policy control steps; row 0 = the post-settle frame the first action
is computed from (so goals already true before the policy acted are visible, and are not counted as events).

  obj_pos     (R, n_obj, 3) float64  movable-object body origins (world, m), names in obj_names
  joint_q     (R, n_joint)  float64  articulated fixture joints (drawer slides m, stove knob rad), joint_names
  goals       (R, n_task)   bool     every suite task's goal predicate, task ids in goal_task_ids
  contact     (R, n_ent)    float64  max robot contact normal force (N) per scene entity (+ "other"), entity_names
  eef_pos     (R, 3)        float64  end-effector position (m)
  gripper_q   (R, 2)        float64  finger joint positions (m)
  targets     JSON string   env.task_targets(t) for every suite task (the classifier needs no simulator)
  schema      int           TRACE_SCHEMA

Sampled once per 20 Hz control step, after the step: contact peaks between physics substeps are not seen.
"""
from __future__ import annotations

import numpy as np

from vlamech.utils.io import from_json, to_json

TRACE_SCHEMA = 1


class TraceRecorder:
    def __init__(self, env):
        self.env = env
        self.obj_names = list(env.movable_objects)
        self.joint_names = list(env.articulated_joints)
        self.entity_names = list(env.contact_entities) + ["other"]
        self.task_ids = sorted(env.goal_status())
        self.rows: dict[str, list] = {k: [] for k in ("obj_pos", "joint_q", "goals", "contact", "eef_pos", "gripper_q")}
        self.last_contacts: dict = {}

    def record(self) -> None:
        e = self.env
        pos = e.object_positions()
        goals = e.goal_status()
        self.last_contacts = e.robot_contacts()
        obs = e._last_obs
        self.rows["obj_pos"].append(np.stack([pos[n] for n in self.obj_names]))
        self.rows["joint_q"].append(e.joint_positions())
        self.rows["goals"].append(np.array([goals[t] for t in self.task_ids], bool))
        self.rows["contact"].append(np.array([self.last_contacts[n] for n in self.entity_names], np.float64))
        self.rows["eef_pos"].append(np.array(obs["robot0_eef_pos"], np.float64))
        self.rows["gripper_q"].append(np.array(obs["robot0_gripper_qpos"], np.float64))

    def arrays(self) -> dict:
        out = {k: np.stack(v) for k, v in self.rows.items()}
        targets = {t: self.env.task_targets(t) for t in self.task_ids}
        out |= {"obj_names": np.array(self.obj_names), "joint_names": np.array(self.joint_names),
                "entity_names": np.array(self.entity_names), "goal_task_ids": np.array(self.task_ids),
                "targets": np.array(to_json(targets)), "schema": np.array(TRACE_SCHEMA)}
        return out


def load_trace(path) -> dict:
    """npz -> dict of arrays, names as lists, targets as {task_id: dict}. Refuses unknown schemas."""
    with np.load(path, allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    if int(t["schema"]) != TRACE_SCHEMA:
        raise ValueError(f"{path}: trace schema {int(t['schema'])}, this code reads {TRACE_SCHEMA}")
    for k in ("obj_names", "joint_names", "entity_names"):
        t[k] = [str(x) for x in t[k]]
    t["goal_task_ids"] = [int(x) for x in t["goal_task_ids"]]
    t["targets"] = {int(k): v for k, v in from_json(str(t["targets"])).items()}
    return t
