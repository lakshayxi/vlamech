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
  delay_<k>       the whole observation of control step t - k at the query at step t (k steps of 0.05 s; frames
                  before the first are the first frame); needs observe() called at every control step (closed_loop does)
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
DELAY_STEPS = (2, 4, 8, 16)  # 0.1, 0.2, 0.4, 0.8 s at 20 Hz
DELAY_FAULTS = tuple(f"delay_{k}" for k in DELAY_STEPS)


class SensorFault:
    def __init__(self, name: str):
        if name not in FAULTS + PHASE_FAULTS + DELAY_FAULTS:
            raise ValueError(f"unknown sensor fault {name!r}; known: {FAULTS + PHASE_FAULTS + DELAY_FAULTS}")
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
        self.frames: dict = {}  # control step -> (agentview, wrist, proprio), filled by observe()

    def observe(self, step, frame, fed, fresh, goals) -> None:
        """frame_hook: record every control step's observation (for delay_<k>)."""
        if self.name.startswith("delay_"):
            self.frames[step] = (frame.agentview, frame.wrist, np.array(frame.proprio, np.float64))
            for old in [t for t in self.frames if t < step - 2 * DELAY_STEPS[-1]]:
                del self.frames[old]

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
        elif self.name.startswith("delay_"):
            step = max(self.frames) + 1 if self.frames else 0  # observe() has seen every step before this query
            k = int(self.name.split("_")[1])
            if step - k >= 0 and (step - k) in self.frames:
                fed, active = list(self.frames[step - k]), True
            elif step > 0:
                fed, active = list(self.frames[min(self.frames)]), True
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


# ------------------------------------------------------------------------------------------------ patching rescue
RESCUE_MODES = ("fresh", "random", "samephase", "otherphase", "noise", "noop")
RESCUES = ("rescue_wrist_12", "rescue_readout_20", "rescue_wrist_12_random", "rescue_readout_20_random",
           "rescue_wrist_12_samephase", "rescue_wrist_12_otherphase", "rescue_wrist_12_noise", "rescue_wrist_12_noop")


class Rescue(SensorFault):
    """lag_wrist (the wrist frame of the previous query) with one layer's states at one token group replaced.

    rescue_<group>_<layer>[_<mode>]; where the replacing states come from (a causal test, not a deployable fix):
      fresh       the same query with the fresh wrist frame (needs the fresh frame)
      random      another episode: same task, next held-out init state, its clean captured observation at the same query
                  index (or its last)
      samephase   the clean captured rollout of the same initial state at the same query index (or its last)
      otherphase  that rollout half an episode away: query (q + n // 2) mod n of its n queries
      noise       the stale states plus Gaussian noise whose norm over the group equals |fresh - stale| there (seeded)
      noop        the stale states themselves (must reproduce lag_wrist exactly)
    Captured donors are clean frames of the held-out clean run (protocol sensor_conflict.captures). Needs the
    in-process policy (hooks)."""

    def __init__(self, name: str):
        if name not in RESCUES:
            raise ValueError(f"unknown rescue {name!r}; known: {RESCUES}")
        parts = name.split("_")
        self.group, self.layer = parts[1], int(parts[2])
        self.mode = parts[3] if len(parts) > 3 else "fresh"
        self.name = name
        self.env, self.key, self.donor_cap = None, None, None
        self.reset()

    def bind(self, env, key: str) -> None:
        self.env, self.key = env, key
        if self.mode in ("random", "samephase", "otherphase"):
            from pathlib import Path

            from vlamech.config import protocol, split_states
            from vlamech.envs.replay import load_capture

            t, init = int(key[4:].split("-")[0]), int(key.split("init")[1])
            states = split_states("confirmatory")
            donor = states[(states.index(init) + 1) % len(states)] if self.mode == "random" else init
            root = Path(__file__).resolve().parents[3]
            self.donor_cap = load_capture(root / protocol("sensor_conflict.captures")["confirmatory"] / "captures"
                                          / f"task{t}-init{donor}.npz")

    def _states(self, policy, obs, instruction):
        from vlamech.interventions.hooks import Capture, hooked

        inp = policy.prepare(*obs, instruction)
        with hooked(policy, captures=[Capture("resid", self.layer, to_cpu=False)], expected_seq_len=inp.layout.seq_len) as s:
            policy.forward(inp)
        return s.captures[("resid", self.layer)]

    def __call__(self, policy, imgs, proprio, instruction):
        from vlamech.interventions.hooks import Intervention, hooked
        from vlamech.interventions.patching import group_positions, noise_patches

        now = (imgs["agentview"], imgs["wrist"], np.array(proprio, np.float64, copy=True))
        fed = (now[0], self.prev[1] if self.prev is not None else now[1], now[2])
        inp = policy.prepare(*fed, instruction)
        pos = group_positions(inp.layout, self.group)
        if self.mode in ("random", "samephase", "otherphase"):
            n = len(self.donor_cap["step"])
            j = (self.q + n // 2) % n if self.mode == "otherphase" else min(self.q, n - 1)
            src = (self.donor_cap["agentview"][j], self.donor_cap["wrist"][j], self.donor_cap["proprio"][j])
        else:
            src = fed if self.mode == "noop" else now
        h = self._states(policy, src, instruction)
        if self.mode == "noise":
            site = ("resid", self.layer)
            stale = self._states(policy, fed, instruction)
            ivs = noise_patches({site: h}, {site: stale}, pos, [self.layer],
                                seed=derive_seed("rescue_noise", self.key, self.q))
        else:
            ivs = [Intervention("resid", self.layer, pos, "replace", h[:, torch_index(pos, h)])]
        with hooked(policy, interventions=ivs, expected_seq_len=inp.layout.seq_len):
            out = policy.forward(inp)
        self.prev = now
        self.q += 1
        return out["actions"], np.asarray(out["actions_norm"]), {"fault_active": self.q > 1}


def torch_index(pos, h):
    import torch

    return torch.as_tensor(np.asarray(pos, np.int64), device=h.device)
