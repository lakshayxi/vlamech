"""Statistics for paired, clustered robot-evaluation data.

Used by: every analysis script (scripts/analysis/).

Data structure assumed throughout: observations are nested  task -> episode (init state) -> step.
Conditions are *paired* on (task, init_state), so contrasts are computed within pairs and uncertainty
comes from resampling init states within each (fixed) task, not from treating steps as independent.

What you can change: number of bootstrap replicates (n_boot), CI level, strata / resampling levels
(the values used in the study come from configs/protocol.yaml -> statistics).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps


def _binary(x, name) -> np.ndarray:
    """Validate a vector of binary outcomes (bool or exactly 0/1); NaN or other values raise."""
    a = np.asarray(x)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if a.dtype == bool:
        return a
    a = a.astype(float)
    if not np.isin(a, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be binary (0/1 or bool); got values {np.unique(a)[:5]}")
    return a.astype(bool)


def _finite(x, name) -> np.ndarray:
    a = np.asarray(x, float)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values: exclude them explicitly (and report the count) first")
    return a


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if not (isinstance(k, (int, np.integer)) and isinstance(n, (int, np.integer)) and 0 <= k <= n and n > 0):
        raise ValueError(f"need integers 0 <= k <= n, n > 0 (got k={k}, n={n})")
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def hierarchical_bootstrap(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    levels: Sequence[str] = ("init_state",),
    strata: Sequence[str] = ("task",),
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI by resampling clusters with replacement, optionally within fixed strata.

    strata: columns whose units are fixed (every stratum appears exactly once in every replicate).
        LIBERO tasks are a fixed set, not a sample from a population of tasks, so the primary analysis
        stratifies by task.
    levels: columns resampled with replacement, outermost first, nested inside the strata. The primary
        unit is the init state: both conditions of a paired comparison share (task, init_state), so a
        pair is always drawn together. Rows below the last level (e.g. steps, conditions) stay intact.
    Sensitivity analysis: strata=(), levels=("task", "init_state") treats tasks as a random sample.
    `statistic` receives the resampled DataFrame and returns a scalar.
    """
    for c in (*strata, *levels):
        if c not in df.columns:
            raise KeyError(f"bootstrap column {c!r} missing from data")
    rng = np.random.default_rng(seed)
    point = float(statistic(df))
    pos = np.arange(len(df))
    if strata:
        skey = df[list(strata)].astype(str).agg("|".join, axis=1).to_numpy()
        units = [_nest(df[skey == k], pos[skey == k], list(levels)) for k in pd.unique(skey)]
    else:
        units = [_nest(df, pos, list(levels))]
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([_draw(u, rng) for u in units])
        reps[b] = statistic(df.iloc[idx])
    n_bad = int((~np.isfinite(reps)).sum())
    if n_bad:
        raise FloatingPointError(f"statistic undefined in {n_bad}/{n_boot} bootstrap replicates; "
                                 "the estimand is not well defined for this data (do not drop replicates silently)")
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot,
            "se": float(np.std(reps, ddof=1)), "strata": list(strata), "levels": list(levels)}


def _nest(sub: pd.DataFrame, pos: np.ndarray, levels: list[str]):
    """Nested lists of positional row indices, one nesting level per resampled column."""
    if not levels:
        return pos
    key = sub[levels[0]].to_numpy()
    return [_nest(sub[key == k], pos[key == k], levels[1:]) for k in pd.unique(key)]


def _draw(groups, rng) -> np.ndarray:
    if isinstance(groups, np.ndarray):
        return groups
    pick = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([_draw(groups[i], rng) for i in pick])


def paired_sign_flip_test(diffs: np.ndarray, n_perm: int = 20_000, seed: int = 0) -> float:
    """Two-sided permutation p-value for mean(diffs) == 0 by random sign flips of paired differences.

    `diffs` must already be aggregated to the independent unit (e.g. one value per episode).
    """
    diffs = _finite(diffs, "diffs")
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))


def task_group_permutation_test(task_values: dict, group: Sequence) -> dict:
    """Exact one-sided test that a named group of tasks has a LOWER mean than the other tasks.

    task_values: {task: value aggregated to the task}. Statistic T = mean(group) - mean(rest); the null distribution
    is T over every way of choosing len(group) tasks (all relabellings, the observed one included), so with 10 tasks
    and a group of 3 the smallest possible p is 1/120. The task is the unit because the claim is about tasks.
    """
    from itertools import combinations

    tasks = sorted(task_values)
    g = set(group)
    if not g < set(tasks):
        raise ValueError(f"group {sorted(g)} is not a proper subset of the tasks {tasks}")
    v = np.array([task_values[t] for t in tasks], float)
    _finite(v, "task_values")

    def stat(members):
        m = np.isin(tasks, list(members))
        return v[m].mean() - v[~m].mean()

    obs = stat(g)
    null = np.array([stat(c) for c in combinations(tasks, len(g))])
    return {"T": float(obs), "p": float((null <= obs + 1e-12).mean()), "n_relabellings": int(len(null))}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test for paired binary outcomes a, b (e.g. success clean vs. corrupted).

    Returns discordant counts and the two-sided binomial p-value, plus the paired risk difference.
    """
    a, b = _binary(a, "a"), _binary(b, "b")
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    n10 = int((a & ~b).sum())  # a succeeds, b fails
    n01 = int((~a & b).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else float(sps.binomtest(min(n10, n01), n, 0.5).pvalue)
    return {"n": int(a.size), "n10": n10, "n01": n01, "p": p, "risk_diff": float(b.mean() - a.mean())}


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as input)."""
    p = _finite(pvals, "pvals")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic; labels: 1 = positive (failure).
    Returns NaN (explicitly undefined) when only one class is present; the bootstrap refuses such replicates."""
    scores, labels = _finite(scores, "scores"), _binary(labels, "labels").astype(int)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have equal shape")
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = sps.rankdata(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))

