"""Which input the policy follows when its inputs describe two different worlds (protocol sensor_conflict.share).

Used by: scripts/run/sensor_conflict.py (input combinations), scripts/analysis/sensor_conflict.py, tests/test_arbitration.py.

For an edit with sources S (e.g. agentview, wrist, proprio), the policy is queried on every mix of the real and the
edited world: subset T of S = the sources taken from the edited world. Combination index = sum of 2**i over the
positions i (in S) of the sources in T, so index 0 is the real world and 2**|S| - 1 the edited one.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

ARM = slice(0, 6)


def subsets(n: int) -> list:
    """Every combination index, as a tuple of 0/1 per source (source i taken from the edited world if 1)."""
    return [tuple((c >> i) & 1 for i in range(n)) for c in range(2 ** n)]


def index(bits) -> int:
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def _arm(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, np.float64)[..., ARM].reshape(*np.shape(a)[:-2], -1)


def values(actions: np.ndarray) -> tuple:
    """actions (2**n, 8, 7) -> (v (2**n,), Delta (48,)): v(T) = <a_T - a_empty, Delta> / |Delta|^2."""
    a = _arm(actions)
    delta = a[-1] - a[0]
    d2 = float(delta @ delta)
    if d2 == 0:
        return np.full(len(a), np.nan), delta
    return (a - a[0]) @ delta / d2, delta


def shapley(v: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley values of the n sources in the game v (indexed by combination); they sum to v[-1] = 1."""
    phi = np.zeros(n)
    for m in range(n):
        others = [i for i in range(n) if i != m]
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for t in combinations(others, k):
                base = sum(1 << i for i in t)
                phi[m] += w * (v[base | (1 << m)] - v[base])
    return phi


def single_source(actions: np.ndarray, m: int) -> dict:
    """For the conflict that takes only source m from the edited world: its value v({m}) and off-axis residual."""
    a = _arm(actions)
    v, delta = values(actions)
    shift = a[1 << m] - a[0]
    r = shift - v[1 << m] * delta
    return {"v": float(v[1 << m]), "residual2": float(r @ r), "delta2": float(delta @ delta), "shift2": float(shift @ shift)}


def gripper_closed(actions: np.ndarray) -> np.ndarray:
    """Per combination: the chunk's gripper decision (chunk mean of the raw [0, 1] output < 0.5 = close)."""
    return np.asarray(actions)[..., 6].mean(-1) < 0.5


def weighted_share(phi: np.ndarray, delta2: np.ndarray) -> np.ndarray:
    """Aggregate per-state shares (n_states, n) with weights |Delta|^2 (protocol sensor_conflict.share)."""
    phi, w = np.asarray(phi, np.float64), np.asarray(delta2, np.float64)
    ok = np.isfinite(phi).all(-1) & (w > 0)
    return (phi[ok] * w[ok, None]).sum(0) / w[ok].sum()
