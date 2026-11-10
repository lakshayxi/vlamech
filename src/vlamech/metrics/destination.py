"""Exploratory destination label (configs/protocol.yaml -> oracle.destination_label).

Used by: scripts/figures_trajectories.py, tests/test_destination.py.

Oracle v2 (vlamech.metrics.outcome) calls a failure intent loss only when another task's goal predicate becomes true
or a non-target object moves first. Carrying the RIGHT object toward ANOTHER task's destination and failing to place
it is therefore labelled execution_loss. This module names the destination an object was heading for:

  destination of task k for object o   mu_k = mean final position of o over clean successes of task k
  approach to k                        1 - min_t |p_o(t) - mu_k| / |p_o(0) - mu_k|      (3-D, fraction closed)
  heading                              argmax_k approach, over the tasks whose manipulated object is o
  counts as another destination only if approach_k > incidental(own, k): the largest approach to k over the clean
  successes of the instructed task (the clean-maximum rule; plain argmax labels nudges of a few percent and the
  drawer passed on the way to the cabinet top)

Only objects manipulated by at least two tasks have a choice of destination (LIBERO-Goal: bowl 1/3/4/8, wine
bottle 2/9). Only episodes with target progress (the v2 event) are labelled: an object that never moved heads nowhere.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def manipulated(trace: dict, task_id: int) -> list:
    return list(trace["targets"][task_id]["manipulated_objects"])


def shared_objects(trace: dict) -> dict:
    """{object: [tasks manipulating it]} for objects manipulated by >= 2 tasks (from the trace's task targets)."""
    by_obj: dict = {}
    for t in trace["goal_task_ids"]:
        for o in manipulated(trace, t):
            by_obj.setdefault(o, []).append(int(t))
    return {o: sorted(ts) for o, ts in by_obj.items() if len(ts) >= 2}


def destinations(clean: Iterable[tuple[int, dict]]) -> dict:
    """{(object, task): mean final position} from (task_id, trace) pairs of clean SUCCESSFUL episodes."""
    finals: dict = {}
    shared = None
    for t, tr in clean:
        shared = shared if shared is not None else shared_objects(tr)
        for o in manipulated(tr, t):
            if o in shared:
                finals.setdefault((o, int(t)), []).append(tr["obj_pos"][-1, tr["obj_names"].index(o)])
    if shared is None:
        raise ValueError("no clean episodes given")
    missing = [(o, t) for o, ts in shared.items() for t in ts if (o, t) not in finals]
    if missing:
        raise ValueError(f"no clean success for destinations {missing}")
    return {k: np.mean(v, axis=0) for k, v in finals.items()}


def incidental(clean: Iterable[tuple[int, dict]], dest: dict) -> dict:
    """{(own task, other task): max approach to the other task's destination over clean successes of own task}."""
    inc: dict = {}
    for t, tr in clean:
        for o in manipulated(tr, t):
            if (o, int(t)) not in dest:
                continue
            path = tr["obj_pos"][:, tr["obj_names"].index(o)]
            for (oo, k), mu in dest.items():
                if oo == o and k != t:
                    inc[(int(t), k)] = max(inc.get((int(t), k), 0.0), approach(path, mu))
    return inc


def approach(path: np.ndarray, mu: np.ndarray) -> float:
    d = np.linalg.norm(np.asarray(path, float) - mu, axis=1)
    if d[0] <= 0:
        raise ValueError("object starts at the destination")
    return float(1.0 - d.min() / d[0])


def heading(trace: dict, task_id: int, dest: dict, progressed: bool, inc: Optional[dict] = None) -> Optional[dict]:
    """{'object', 'heading', 'approach': {task: fraction}} or None (no shared object, or no target progress).
    With `inc` (incidental()), another task is the heading only if its approach also beats the clean maximum."""
    if not progressed:
        return None
    objs = [o for o in manipulated(trace, task_id) if any(k[0] == o for k in dest)]
    if len(objs) != 1:
        return None
    o = objs[0]
    path = trace["obj_pos"][:, trace["obj_names"].index(o)]
    app = {t: approach(path, mu) for (oo, t), mu in sorted(dest.items()) if oo == o}
    # another task qualifies if the object got closer to it (relatively) than to its own destination and, with `inc`,
    # farther toward it than any clean success of the instructed task goes on its way
    other = [k for k in app if k != task_id and app[k] > app[task_id]
             and (inc is None or app[k] > inc.get((task_id, k), 0.0))]
    best = max(other, key=lambda k: app[k]) if other else task_id
    return {"object": o, "heading": best, "approach": app}


def label(outcome_v2: str, task_id: int, head: Optional[dict]) -> str:
    """v2 outcome, except a failure heading for another task's destination is 'wrong_destination'."""
    if outcome_v2 != "success" and head is not None and head["heading"] != task_id:
        return "wrong_destination"
    return outcome_v2
