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
  lag_wrist_wait  lag_wrist, but the robot holds still for one chunk (8 steps) before every query, so the late wrist
                  frame shows the scene as it is now; a recovery that uses only frames that have arrived. Holding
                  doubles the time, so the episode gets twice the step budget (step_factor)
  delay_8_wait    the same for delay_8: every source 0.4 s late, the robot holds still for 8 steps before every query
Mitigations of the late wrist camera that use only data that has arrived (the wrist frame of the previous query; fresh
proprio and the camera pose that forward kinematics gives from it):
  lag_wrist_waitopen  hold still before a query only while the fresh finger reading says the gripper is open (before the
                      grasp); move freely once it has closed
  lag_wrist_slow      every commanded motion halved, so the late frame is half as far behind; twice the step budget
  lag_wrist_warp      the late wrist frame reprojected to the current camera pose, assuming every pixel lies on the
                      table plane; the gripper's own pixels (assets/wrist_gripper_mask.npz) are kept as they are
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
FAULTS = ("none", "proprio_offset", "gripper_stuck", "lag_agentview", "lag_wrist", "lag_proprio", "lag_all",
          "lag_wrist_wait", "lag_wrist_waitopen", "lag_wrist_slow", "lag_wrist_warp")
WRIST_LAGS = ("lag_wrist", "lag_wrist_wait", "lag_wrist_waitopen", "lag_wrist_slow", "lag_wrist_warp")
OPEN_ABOVE = 0.04  # finger gap (m) above which the gripper counts as open
SLOW = 0.5
TABLE_Z = 0.92  # table plane for the warp (m); fitted on calibration frames of random hand motion
WRIST_CAM = "robot0_eye_in_hand"
PHASE_FAULTS = ("freeze_approach", "freeze_grasp", "freeze_transport", "freeze_place")
DELAY_STEPS = (2, 4, 8, 16)  # 0.1, 0.2, 0.4, 0.8 s at 20 Hz
DELAY_FAULTS = tuple(f"delay_{k}" for k in DELAY_STEPS) + ("delay_8_wait",)
WAIT_FAULTS = ("lag_wrist_wait", "delay_8_wait", "lag_wrist_waitopen", "lag_wrist_slow")


class SensorFault:
    def __init__(self, name: str):
        if name not in FAULTS + PHASE_FAULTS + DELAY_FAULTS:
            raise ValueError(f"unknown sensor fault {name!r}; known: {FAULTS + PHASE_FAULTS + DELAY_FAULTS}")
        self.name = name
        self.step_factor = 2 if name in WAIT_FAULTS else 1
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
        self.last_action: Optional[np.ndarray] = None
        self.held_last = False
        self.prev_pose: Optional[tuple] = None

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
        hold = self.q % 2 == 1 if self.name in ("lag_wrist_wait", "delay_8_wait") else (
            self.name == "lag_wrist_waitopen" and self.q > 0 and not self.held_last and now[2][6] - now[2][7] > OPEN_ABOVE)
        pose = self._wrist_pose() if self.name == "lag_wrist_warp" else None
        if hold:  # hold still: zero motion, keep the gripper command
            acts = np.zeros((8, 7))
            acts[:, 6] = self.last_action[6]
            self.prev, self.held_last = now, True
            self.q += 1
            return acts, np.zeros((8, 7), np.float32), {"fault_active": True, "hold": True}
        self.held_last = False
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
            which = SOURCES if self.name == "lag_all" else ("wrist",) if self.name in WRIST_LAGS else (self.name[4:],)
            for s in which:
                fed[SOURCES.index(s)] = prev[SOURCES.index(s)]
            active = self.prev is not None
            if self.name == "lag_wrist_warp" and self.prev_pose is not None:
                fed[1] = warp_to(prev[1], self.prev_pose, pose)
        elif self.name.startswith("delay_"):
            step = max(self.frames) + 1 if self.frames else 0  # observe() has seen every step before this query
            k = int(self.name.split("_")[1])  # delay_8_wait: 8
            if step - k >= 0 and (step - k) in self.frames:
                fed, active = list(self.frames[step - k]), True
            elif step > 0:
                fed, active = list(self.frames[min(self.frames)]), True
        elif self.name in PHASE_FAULTS and self.held is not None and self.q <= self.trigger_q + FREEZE_QUERIES:
            fed, active = list(self.held), True
        out = policy.forward(policy.prepare(fed[0], fed[1], fed[2], instruction))
        if self.name == "lag_wrist_slow":
            out = dict(out, actions=np.asarray(out["actions"], np.float64).copy())
            out["actions"][:, :6] *= SLOW
        self.prev_pose = pose
        norm = np.asarray(out["actions_norm"])
        if self.grasp_q is None and norm[:, 6].mean() < GRIP_CLOSED_BELOW:
            self.grasp_q = self.q
        if self.name in PHASE_FAULTS and self.trigger_q is None and self._triggered():
            self.trigger_q, self.held = self.q, now
        self.last_action = np.asarray(out["actions"])[-1].copy()
        info = {"fault_active": bool(active)}
        if self.name in PHASE_FAULTS:
            info["trigger_query"] = self.trigger_q
        self.prev = now
        self.q += 1
        return out["actions"], norm, info


    def _wrist_pose(self):
        sim = self.env._env.sim
        cid = sim.model.camera_name2id(WRIST_CAM)
        return (sim.data.cam_xpos[cid].copy(), sim.data.cam_xmat[cid].reshape(3, 3).copy(), float(sim.model.cam_fovy[cid]))


_MASK = None


def gripper_mask() -> np.ndarray:
    global _MASK
    if _MASK is None:
        from pathlib import Path

        _MASK = np.load(Path(__file__).resolve().parents[3] / "assets" / "wrist_gripper_mask.npz")["mask"]
    return _MASK


def warp_to(img: np.ndarray, old: tuple, new: tuple, z0: float = TABLE_Z) -> np.ndarray:
    """The wrist image taken at camera pose `old` as it would look from pose `new` if the scene were the table plane
    z = z0 (nearest-pixel backward warp). Policy images are mirrored left-right against MuJoCo's camera frame. Pixels
    that fall outside the old view, and the gripper's own pixels, keep their old values."""
    (p0, m0, fovy), (p1, m1, _) = old, new
    h, w = img.shape[:2]
    f, c = (h / 2) / np.tan(np.deg2rad(fovy) / 2), (w - 1) / 2
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    u = w - 1 - u
    d = np.stack([(u - c) / f, -(v - c) / f, -np.ones_like(u)], -1) @ m1.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z0 - p1[2]) / d[..., 2]
        pc = (p1 + t[..., None] * d - p0) @ m0
        us = w - 1 - (c + f * pc[..., 0] / -pc[..., 2])
        vs = c - f * pc[..., 1] / -pc[..., 2]
    ok = (t > 0) & (pc[..., 2] < 0) & (us >= 0) & (us <= w - 1) & (vs >= 0) & (vs <= h - 1)
    mask = gripper_mask()
    if mask.shape == (h, w):
        ok &= ~mask
    out = np.array(img, copy=True)
    out[ok] = img[np.rint(vs[ok]).astype(int), np.rint(us[ok]).astype(int)]
    return out


# ------------------------------------------------------------------------------------------------ patching rescue
RESCUE_MODES = ("fresh", "random", "samephase", "otherphase", "noise", "noop")
RESCUES = ("rescue_wrist_12", "rescue_readout_20", "rescue_wrist_12_random", "rescue_readout_20_random",
           "rescue_wrist_12_samephase", "rescue_wrist_12_otherphase", "rescue_wrist_12_noise", "rescue_wrist_12_noop")
SWEEP_LAYERS = (4, 8, 16, 20, 24, 28)  # layer sweep of the fresh and random wrist rescues (exploratory follow-up)
RESCUES += tuple(f"rescue_wrist_{l}{m}" for l in SWEEP_LAYERS for m in ("", "_random"))


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
