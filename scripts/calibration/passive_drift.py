"""Passive drift: how far objects and fixture joints move with no robot motion (oracle floor, classifier v2).

Usage:  python scripts/calibration/passive_drift.py [--workers 10]  ->  results/oracle_calibration/passive_drift.json

After the settle steps objects keep settling by a deterministic sub-micron amount (e.g. the bowl 0.279 um). Where a
task's displacement threshold is itself at that scale, the target object's own settling would count as "moved". The
floor is a property of the simulator and the initial state, not of the policy, so it is measured for every
benchmark init state of every task over the episode's full length: reset as in every episode, then max_steps steps
of upstream's no-op action (zero arm motion, gripper open).
floor[task-init][object] = max over the steps of |p - p_post-settle| for that task and init state (passive drift
differs between states: one task-8 state lets the bowl slide about 1.4 mm with no robot action, all others settle by
< 0.3 um); floor[task-init][joint] likewise for |q - q0|. invalid[task-init]: with no robot action an object falls
off the table or moves more than its own size (none in LIBERO-Goal).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "oracle_calibration" / "passive_drift.json"


def bounds3d(sim, prefix: str) -> np.ndarray:
    """[[xmin, ymin, zmin], [xmax, ymax, zmax]] of every visual/collision geom under bodies named prefix*."""
    m = sim.model
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        b, ok = int(m.geom_bodyid[g]), False
        while b > 0:
            if m.body_id2name(b).startswith(prefix):
                ok = True
                break
            b = int(m.body_parentid[b])
        if not ok or m.geom_group[g] not in (0, 1):
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
        w = corners @ sim.data.geom_xmat[g].reshape(3, 3).T + sim.data.geom_xpos[g]
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    if not np.isfinite(lo).all():
        raise KeyError(f"no geoms under {prefix}")
    return np.stack([lo, hi])


def measure_task(args) -> dict:
    from vlamech.config import protocol
    from vlamech.envs.libero_env import DUMMY_ACTION, LiberoTaskEnv

    suite, task_id, states, steps = args
    env = LiberoTaskEnv(suite, task_id, resolution=int(protocol("env.resolution")),
                        settle_steps=int(protocol("env.settle_steps")), env_seed=int(protocol("env.env_seed")),
                        render=False)
    per = {}
    for s in states:
        env.reset(s)
        p0, q0 = env.object_positions(), env.joint_positions()
        size = {}
        for n in env.movable_objects:  # largest side of the object's world bounding box at the post-settle frame
            b = bounds3d(env.inner.sim, env.inner.objects_dict[n].root_body)
            size[n] = float((b[1] - b[0]).max())
        obj = {n: 0.0 for n in env.movable_objects}
        jnt = {n: 0.0 for n in env.articulated_joints}
        fell = set()
        for _ in range(steps):
            env._env.step(DUMMY_ACTION.tolist())  # env-space no-op, exactly the settle action
            for n, p in env.object_positions().items():
                obj[n] = max(obj[n], float(np.linalg.norm(p - p0[n])))
                if env.off_table(p):
                    fell.add(n)
            for n, d in zip(env.articulated_joints, np.abs(env.joint_positions() - q0)):
                jnt[n] = max(jnt[n], float(d))
        moved = sorted(n for n in obj if obj[n] > size[n])
        per[f"{task_id}-{s}"] = {"objects": obj, "joints": jnt, "size": size, "fell_off_table": sorted(fell),
                                 "moved_more_than_own_size": moved, "invalid": bool(fell or moved)}
    env.close()
    return per


def main():
    from vlamech.config import protocol
    from vlamech.utils.io import atomic_write_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10, help="parallel simulator processes")
    args = ap.parse_args()
    suite, steps = protocol("env.suite"), int(protocol("env.max_steps"))
    jobs = [(suite, t, list(range(int(protocol("splits.n_init_states")))), steps) for t in range(10)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        per = {k: v for r in ex.map(measure_task, jobs) for k, v in r.items()}
    out = {"rule": "per (task, init state): max over max_steps no-op steps after reset of |p - p_post-settle| "
                   "(objects, m) and |q - q0| (joints)",
           "suite": suite, "max_steps": steps, "floor": per}
    atomic_write_json(OUT, out)  # saved before any summary: a summary bug must not lose the measurement
    worst = {}
    for v in per.values():
        for n, x in v["objects"].items():
            worst[n] = max(worst.get(n, 0.0), x)
    big = {k: {n: f"{x:.3g}" for n, x in v["objects"].items() if x > 1e-6} for k, v in per.items()}
    invalid = sorted(k for k, v in per.items() if v["invalid"])
    print(f"invalid (with no robot action an object falls off the table or moves more than its own size): "
          f"{len(invalid)}/{len(per)}", invalid)
    print("max per object:", {k: f"{v:.3g}" for k, v in worst.items()}, "| episodes above 1 um:", {k: v for k, v in big.items() if v})


if __name__ == "__main__":
    main()
