"""Record the fixture layout that upstream evaluation uses for every (task, init state).

Usage:  python scripts/setup/build_fixture_layouts.py   -> assets/fixture_layouts_libero_goal.npz

Why: LIBERO re-samples the placement of fixed bodies
(cabinet, stove, wine rack) on every env.reset(). Their positions live in the MuJoCo *model* (body_pos /
body_quat), not in the simulator state, so `set_init_state` does not restore them. Upstream evaluation
(run_libero_eval.py) creates one env per task, seeds it with 0, and runs init states 0..49 in order, so the
episode for init state k always sees the layout of the (k+1)-th reset. Any other order (shards, resumed runs,
different conditions) would see different fixture positions. We replay upstream's exact sequence once, store
the pose of every body whose pose varies, and LiberoTaskEnv.reset(k) restores it, which makes every episode
identical to upstream's episode k regardless of run order. Verified bit-exact by tests/test_libero_env.py.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "assets" / "fixture_layouts_libero_goal.npz"


def replay_task(args) -> dict:
    """Worker wrapper: exceptions are returned as text (simulator exceptions do not pickle across processes)."""
    import traceback

    try:
        return _replay_task(*args)
    except Exception:
        raise RuntimeError(f"task {args}:\n{traceback.format_exc()}") from None


def _replay_task(suite_name: str, task_id: int) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    from vlamech.config import protocol

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    res = int(protocol("env.resolution"))
    # no renderer: placement sampling does not depend on rendering (checked: the LIBERO-Goal file is reproduced
    # bit-exactly, tests/test_libero_env.py), and render contexts of 50 hard resets exhaust GPU memory under load
    env = ControlEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False)
    env.seed(int(protocol("env.env_seed")))  # exactly as upstream get_libero_env
    inits = suite.get_task_init_states(task_id)
    pos, quat = [], []
    for k in range(len(inits)):  # upstream order: episode k is reset number k+1
        env.reset()
        env.set_init_state(inits[k])
        m = env.env.sim.model
        pos.append(np.array(m.body_pos, copy=True))
        quat.append(np.array(m.body_quat, copy=True))
    names = [env.env.sim.model.body_id2name(i) for i in range(env.env.sim.model.nbody)]
    env.close()
    return {"task_id": task_id, "names": names, "pos": np.stack(pos), "quat": np.stack(quat)}


def main():
    import argparse

    from vlamech.config import protocol
    from vlamech.utils.io import atomic_write_npz

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10, help="parallel simulator processes")
    args = ap.parse_args()
    suite = protocol("env.suite")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = sorted(ex.map(replay_task, [(suite, t) for t in range(10)]), key=lambda r: r["task_id"])
    names = results[0]["names"]  # one file: every LIBERO-Goal task has the same bodies (checked)
    if any(r["names"] != names for r in results):
        raise RuntimeError("body lists differ across tasks")
    pos = np.stack([r["pos"] for r in results])    # (task, init, body, 3)
    quat = np.stack([r["quat"] for r in results])  # (task, init, body, 4)
    varying = np.nonzero((np.ptp(pos, axis=(0, 1)).max(-1) > 0) | (np.ptp(quat, axis=(0, 1)).max(-1) > 0))[0]
    print("bodies whose pose varies across resets:", [names[i] for i in varying])
    atomic_write_npz(OUT, body_names=np.array([names[i] for i in varying]), body_pos=pos[:, :, varying],
                     body_quat=quat[:, :, varying])
    print(f"wrote {OUT.relative_to(ROOT)}  shape pos {pos[:, :, varying].shape}")


if __name__ == "__main__":
    main()
