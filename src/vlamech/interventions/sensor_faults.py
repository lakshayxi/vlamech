"""Closed-loop sensor faults: what the policy is given differs from what the robot's sensors read (protocol
sensor_conflict.faults and sensor_conflict.phase_faults).

Used by: scripts/run/closed_loop.py (--sensor-fault), tests/test_sensor_faults.py.

A fault is an action function (vlamech.envs.episode.run_episode protocol): it receives the observation of the query
(both camera frames after any corruption, and the proprio), changes what the policy is fed, and queries the plain
policy. It holds per-episode state: run_episode calls reset(), and closed_loop calls bind(env, episode key) first,
because the phase triggers read object and hand positions from the simulator (never from what the policy sees).
  proprio_offset  reported end-effector position off by 3 cm in a horizontal direction seeded per episode
  gripper_stuck   reported finger positions held at the first query's values (open after the settle steps)
  lag_<source>    that source from the previous query (agentview, wrist, proprio, or all of them)
  freeze_<phase>  the trigger query's whole observation fed again at the next two queries. Triggers, checked after each
                  query: approach = hand within 15 cm of the object to grasp; grasp = the query whose chunk first commands
                  closing; transport = three queries later; place = after the grasp, the object within 15 cm (horizontal)
                  of the goal region (the second argument of the task's goal predicate)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vlamech.perturbations.worlds import EEF_SITE
from vlamech.utils.seeding import derive_seed

SOURCES = ("agentview", "wrist", "proprio")
OFFSET_M = 0.03
NEAR_M = 0.15
FREEZE_QUERIES = 2
GRIP_CLOSED_BELOW = 0.5
FAULTS = ("none", "proprio_offset", "gripper_stuck", "lag_agentview", "lag_wrist", "lag_proprio", "lag_all")
PHASE_FAULTS = ("freeze_approach", "freeze_grasp", "freeze_transport", "freeze_place")


class SensorFault:
    def __init__(self, name: str):
        if name not in FAULTS + PHASE_FAULTS:
            raise ValueError(f"unknown sensor fault {name!r}; known: {FAULTS + PHASE_FAULTS}")
        self.name = name
        self.env, self.key = None, None
        self.reset()

    def bind(self, env, key: str) -> None:
        self.env, self.key = env, key
        t = env.task_targets(env.task_id)
        self.manip = (t["manipulated_objects"] or [None])[0]
        goal = [a for pred, *a in env._parsed[env.task_id]["goal_state"] if len(a) == 2 and a[0] == self.manip]
        self.dest = goal[0][1] if goal else None
        if self.name in PHASE_FAULTS and (self.manip is None or self.dest is None):
            raise ValueError(f"{self.name} needs a pick-and-place task (task {env.task_id})")
        ang = np.random.default_rng(derive_seed("sensor_fault", "offset", key)).integers(8) * np.pi / 4
        self.offset = OFFSET_M * np.array([np.cos(ang), np.sin(ang), 0.0])

    def reset(self) -> None:
        self.q = 0
        self.prev: Optional[tuple] = None
        self.first_proprio: Optional[np.ndarray] = None
        self.grasp_q: Optional[int] = None
        self.trigger_q: Optional[int] = None
        self.held: Optional[tuple] = None

    # ------------------------------------------------------------------ phase triggers (simulator positions)
    def _pos(self, entity: str) -> np.ndarray:
        """A movable object's position, or a goal region's site position."""
        if entity in self.env.movable_objects:
            return self.env.object_positions()[entity]
        sim = self.env.inner.sim
        return np.array(sim.data.site_xpos[sim.model.site_name2id(entity)])

    def _eef(self) -> np.ndarray:
        sim = self.env.inner.sim
        return np.array(sim.data.site_xpos[sim.model.site_name2id(EEF_SITE)])

    def _triggered(self) -> bool:
        phase = self.name.split("_", 1)[1]
        if phase == "approach":
            return bool(np.linalg.norm(self._eef() - self._pos(self.manip)) < NEAR_M)
        if self.grasp_q is None:
            return False
        if phase == "grasp":
            return self.q == self.grasp_q
        if phase == "transport":
            return self.q == self.grasp_q + 3
        d = self._pos(self.manip)[:2] - self._pos(self.dest)[:2]
        return self.q >= self.grasp_q and bool(np.linalg.norm(d) < NEAR_M)

    # ------------------------------------------------------------------ action function
    def __call__(self, policy, imgs, proprio, instruction):
        now = (imgs["agentview"], imgs["wrist"], np.array(proprio, np.float64, copy=True))
        if self.first_proprio is None:
            self.first_proprio = now[2].copy()
        fed = list(now)
        active = False
        if self.name == "proprio_offset":
            fed[2] = now[2].copy()
            fed[2][:3] += self.offset
            active = True
        elif self.name == "gripper_stuck":
            fed[2] = now[2].copy()
            fed[2][6:8] = self.first_proprio[6:8]
            active = True
        elif self.name.startswith("lag_"):
            prev = self.prev if self.prev is not None else now
            which = SOURCES if self.name == "lag_all" else (self.name[4:],)
            for s in which:
                fed[SOURCES.index(s)] = prev[SOURCES.index(s)]
            active = self.prev is not None
        elif self.name in PHASE_FAULTS and self.held is not None and self.q <= self.trigger_q + FREEZE_QUERIES:
            fed, active = list(self.held), True
        out = policy.forward(policy.prepare(fed[0], fed[1], fed[2], instruction))
        norm = np.asarray(out["actions_norm"])
        if self.grasp_q is None and norm[:, 6].mean() < GRIP_CLOSED_BELOW:
            self.grasp_q = self.q
        if self.name in PHASE_FAULTS and self.trigger_q is None and self._triggered():
            self.trigger_q, self.held = self.q, now
        info = {"fault_active": bool(active)}
        if self.name in PHASE_FAULTS:
            info["trigger_query"] = self.trigger_q
        self.prev = now
        self.q += 1
        return out["actions"], norm, info
