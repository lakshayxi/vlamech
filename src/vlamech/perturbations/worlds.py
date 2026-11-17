"""Counterfactual worlds: the scene rendered as it would look if one physical fact were different.

Used by: scripts/run/sensor_conflict.py (open-loop sensor-conflict pass), tests/test_worlds.py.

A world is a simulator state (and, for fixtures, a temporary change to a body position in the model). It is only
rendered, never simulated from: render() puts the env into the edited state, regenerates the observation, and then
undoes every model change. The three sources the policy reads (third-person frame, wrist frame, proprio) all come
from the same render, so a world is internally consistent; mixing sources from two worlds is what creates a conflict.

Edits (protocol sensor_conflict.edits):
  shift_entity  move a movable object (its free joint) or a fixture (its root body in the model) by a horizontal vector
  shift_arm     move the end effector by a horizontal vector at fixed orientation (damped least squares on the arm
                joints); a held object moves with the hand
  flip_gripper  open a closed gripper or close an open one (finger joints)
Every edit is checked for new interpenetration (contacts deeper than protocol sensor_conflict.max_penetration_m that
were not there before); directions are tried in a seeded order and the first valid one is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

EEF_SITE = "gripper0_grip_site"
FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
FINGER_OPEN = (0.04, -0.04)
FINGER_CLOSED = (0.0005, -0.0005)


@dataclass
class World:
    """An edited simulator state plus the model body offsets it needs (applied only while rendering)."""

    state: np.ndarray
    body_offsets: dict = field(default_factory=dict)  # body id -> (3,) offset added to model.body_pos
    edit: str = "none"
    detail: dict = field(default_factory=dict)


def _sim(env):
    return env.inner.sim


def _load(env, world: World):
    sim = _sim(env)
    saved = {}
    for b, off in world.body_offsets.items():
        saved[b] = np.array(sim.model.body_pos[b], copy=True)
        sim.model.body_pos[b] = saved[b] + off
    sim.set_state_from_flattened(world.state)
    sim.forward()
    return saved


def _unload(env, saved: dict):
    sim = _sim(env)
    for b, pos in saved.items():
        sim.model.body_pos[b] = pos
    sim.forward()


def render(env, world: World):
    """Frame (agentview, wrist, proprio) of a world. Model changes are undone before returning."""
    sim = _sim(env)
    saved = {}
    for b, off in world.body_offsets.items():
        saved[b] = np.array(sim.model.body_pos[b], copy=True)
        sim.model.body_pos[b] = saved[b] + off
    try:
        frame = env.set_sim_state(world.state)
    finally:
        _unload(env, saved)
    return frame


def penetrations(env, world: World, depth: float) -> set:
    """Geom pairs in contact deeper than `depth` (m) in a world."""
    saved = _load(env, world)
    try:
        d = _sim(env).data
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < -depth:
                out.add((min(c.geom1, c.geom2), max(c.geom1, c.geom2)))
        return out
    finally:
        _unload(env, saved)


def directions(n: int, seed: int) -> np.ndarray:
    """n horizontal unit vectors at equal angles, in a seeded random order."""
    ang = 2 * np.pi * np.arange(n) / n
    vec = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1)
    return vec[np.random.default_rng(seed).permutation(n)]


# ------------------------------------------------------------------------------------------------------------ edits
def _qpos_slice(sim, joint: str) -> slice:
    addr = sim.model.get_joint_qpos_addr(joint)  # an int for 1-dof joints, (start, end) for free joints
    return slice(*addr) if isinstance(addr, tuple) else slice(int(addr), int(addr) + 1)


def _flat(sim) -> np.ndarray:
    return np.array(sim.get_state().flatten(), copy=True)


def shift_entity(env, base: World, name: str, delta: np.ndarray) -> World:
    """Move a movable object or a fixture by `delta` (m, world frame)."""
    sim = _sim(env)
    saved = _load(env, base)
    try:
        if name in env.movable_objects:
            sl = _qpos_slice(sim, f"{name}_joint0")
            sim.data.qpos[sl.start:sl.start + 3] += delta
            sim.forward()
            return World(_flat(sim), dict(base.body_offsets), "shift_entity", {"entity": name, "delta": delta.tolist()})
        body = sim.model.body_name2id(f"{name}_main")
        offs = dict(base.body_offsets)
        offs[body] = offs.get(body, np.zeros(3)) + delta
        return World(np.array(base.state, copy=True), offs, "shift_entity", {"entity": name, "delta": delta.tolist()})
    finally:
        _unload(env, saved)


def _arm_dofs(sim, env):
    joints = env.inner.robots[0].robot_joints
    return [sim.model.get_joint_qvel_addr(j) for j in joints], [sim.model.get_joint_qpos_addr(j) for j in joints]


def _rot_err(r_target: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Axis-angle vector of r_target r^T (small-angle orientation error)."""
    m = r_target @ r.T
    return 0.5 * np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])


def shift_arm(env, base: World, delta: np.ndarray, carry: Optional[str] = None, iters: int = 50,
              tol: float = 2e-4) -> World:
    """Move the end-effector site by `delta` at fixed orientation; `carry` = a held object moved with the hand."""
    sim = _sim(env)
    saved = _load(env, base)
    try:
        sid = sim.model.site_name2id(EEF_SITE)
        dofs, qaddr = _arm_dofs(sim, env)
        p0 = np.array(sim.data.site_xpos[sid], copy=True)
        r0 = np.array(sim.data.site_xmat[sid].reshape(3, 3), copy=True)
        target = p0 + delta
        lo = sim.model.jnt_range[[sim.model.joint_name2id(j) for j in env.inner.robots[0].robot_joints], 0]
        hi = sim.model.jnt_range[[sim.model.joint_name2id(j) for j in env.inner.robots[0].robot_joints], 1]
        err_norm = np.inf
        for _ in range(iters):
            p = sim.data.site_xpos[sid]
            r = sim.data.site_xmat[sid].reshape(3, 3)
            err = np.concatenate([target - p, _rot_err(r0, r)])
            err_norm = float(np.linalg.norm(err[:3]))
            if err_norm < tol and np.linalg.norm(err[3:]) < 1e-3:
                break
            jp = sim.data.get_site_jacp(EEF_SITE).reshape(3, -1)[:, dofs]
            jr = sim.data.get_site_jacr(EEF_SITE).reshape(3, -1)[:, dofs]
            j = np.vstack([jp, jr])
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), err)
            q = sim.data.qpos[qaddr] + dq
            sim.data.qpos[qaddr] = np.clip(q, lo, hi)
            sim.forward()
        reached = float(np.linalg.norm(sim.data.site_xpos[sid] - target))
        if carry is not None:
            sl = _qpos_slice(sim, f"{carry}_joint0")
            sim.data.qpos[sl.start:sl.start + 3] += delta
            sim.forward()
        return World(_flat(sim), dict(base.body_offsets), "shift_arm",
                     {"delta": delta.tolist(), "ik_error_m": reached, "carry": carry})
    finally:
        _unload(env, saved)


def gripper_opening(env, base: World) -> float:
    sim = _sim(env)
    saved = _load(env, base)
    try:
        q = [float(sim.data.qpos[_qpos_slice(sim, j).start]) for j in FINGER_JOINTS]
        return q[0] - q[1]
    finally:
        _unload(env, saved)


def flip_gripper(env, base: World, open_above: float) -> World:
    """Close an open gripper (opening > open_above, m) or open a closed one."""
    sim = _sim(env)
    saved = _load(env, base)
    try:
        q = [float(sim.data.qpos[_qpos_slice(sim, j).start]) for j in FINGER_JOINTS]
        was_open = (q[0] - q[1]) > open_above
        new = FINGER_CLOSED if was_open else FINGER_OPEN
        for j, v in zip(FINGER_JOINTS, new):
            sim.data.qpos[_qpos_slice(sim, j).start] = v
        sim.forward()
        return World(_flat(sim), dict(base.body_offsets), "flip_gripper",
                     {"from": "open" if was_open else "closed", "opening_m": q[0] - q[1]})
    finally:
        _unload(env, saved)


def first_valid(env, base: World, make, dirs: np.ndarray, depth: float):
    """The first world make(direction) that adds no interpenetration deeper than `depth`, or None."""
    before = penetrations(env, base, depth)
    for d in dirs:
        w = make(d)
        if w is not None and not (penetrations(env, w, depth) - before):
            return w
    return None
