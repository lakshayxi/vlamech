"""Online failure signals for the failure monitor and the safe stop (protocol mitigation.monitor_*).

Used by: scripts/analysis/failure_monitor.py. Every signal at query q uses only commanded actions up to and including q, so it is
available before chunk q is executed.
  instruction_effect  |a(o, l) - a(o, empty)| over the normalized 8x6 arm chunk (one extra forward)
  chunk_turn          degrees between the last commanded translation of chunk q-1 and the first of chunk q
  horizon_decay       log(first-step speed / last-step speed) of chunk q: positive when the chunk slows down
  chunk_jump          distance between the last commanded arm step of chunk q-1 and the first of chunk q
  speed               mean commanded translation speed of chunk q
Undefined values (the first query has no previous chunk) are NaN and ignored by the episode maximum.
"""
from __future__ import annotations

import math

import numpy as np

SIGNALS = ("instruction_effect", "chunk_turn", "horizon_decay", "chunk_jump", "speed")
_SPEED_FLOOR = 1e-6  # definitional: keeps the log finite when a commanded step is exactly zero


def query_signals(actions_norm: np.ndarray, instruction_effect: np.ndarray) -> dict:
    """{signal: array of length Q} from (Q, 8, 7) commanded normalized actions and the per-query instruction effect."""
    a = np.asarray(actions_norm, np.float64)[:, :, :6]
    ie = np.asarray(instruction_effect, np.float64)
    if a.ndim != 3 or a.shape[1] != 8 or len(ie) != len(a):
        raise ValueError(f"expected (Q, 8, 6+) actions and Q effects, got {a.shape} and {ie.shape}")
    t = a[:, :, :3]
    sp = np.linalg.norm(t, axis=-1)
    turn, jump = np.full(len(a), np.nan), np.full(len(a), np.nan)
    for q in range(1, len(a)):
        u, v = t[q - 1, -1], t[q, 0]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 0 and nv > 0:
            turn[q] = math.degrees(math.acos(float(np.clip(u @ v / (nu * nv), -1.0, 1.0))))
        jump[q] = float(np.linalg.norm(a[q, 0] - a[q - 1, -1]))
    decay = np.log(np.maximum(sp[:, 0], _SPEED_FLOOR) / np.maximum(sp[:, -1], _SPEED_FLOOR))
    return {"instruction_effect": ie, "chunk_turn": turn, "horizon_decay": decay, "chunk_jump": jump, "speed": sp.mean(1)}


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve, step form); labels: 1 = positive (failure)."""
    s, y = np.asarray(scores, np.float64), np.asarray(labels, bool)
    if not y.any():
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    hits = np.cumsum(y[order])
    precision = hits / np.arange(1, len(s) + 1)
    return float(precision[y[order]].mean())


def conformal_threshold(episode_scores: np.ndarray, alpha: float) -> float:
    """Split conformal: the ceil((N + 1)(1 - alpha))-th smallest episode maximum; +inf when that rank exceeds N."""
    s = np.sort(np.asarray(episode_scores, np.float64))
    if len(s) == 0 or not np.isfinite(s).all():
        raise ValueError("calibration scores must be non-empty and finite")
    rank = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if rank > len(s) else float(s[rank - 1])


def first_exceedance(z: np.ndarray, threshold: float):
    """Index of the first query whose score exceeds the threshold, or None (NaN never exceeds)."""
    with np.errstate(invalid="ignore"):
        hits = np.nonzero(np.asarray(z, np.float64) > threshold)[0]
    return int(hits[0]) if len(hits) else None
