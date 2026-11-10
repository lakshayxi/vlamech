"""Severity selection (protocol corruption.selection_rule, corruption.targets).

Used by: scripts/analysis/severity_calibration.py. Pure functions, tested in tests/test_selection.py.

Rule, per family: isotonic (monotone non-increasing) regression of success on ladder index, weighted by episodes
(pool-adjacent-violators); relative success = fitted / clean success. For each target r, the level whose fitted
relative success is closest to r (ties -> the milder level). If even the top level stays above the lowest target,
the ladder is extended by one step of its own spacing, once, if the parameter's physical bound allows it;
otherwise (or if still above after the extension) the family is "benign at all tested severities" and kept only at
its top level. A target the top level cannot reach is recorded as unreachable and gets no level; "benign" means no
target is reachable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

TIE_ATOL = 1e-12  # numerical: rates are multiples of 1/50, far coarser than this


def isotonic_nonincreasing(y, w) -> np.ndarray:
    """Weighted least-squares fit constrained to be non-increasing (pool adjacent violators)."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    if y.shape != w.shape or y.ndim != 1 or not len(y) or (w <= 0).any() or not np.isfinite(y).all():
        raise ValueError("isotonic: need equal-length finite y and positive weights")
    blocks = []  # [mean, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:  # violation of non-increasing
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([np.full(c, m) for m, _, c in blocks])


def next_level(ladder: list, kind: str, bound: Optional[float]) -> Optional[float]:
    """One more step of the ladder's own spacing ("linear" or "geometric"), or None past the physical bound.
    For decreasing ladders (JPEG quality) the step continues downwards."""
    a, b = ladder[-2], ladder[-1]
    nxt = b + (b - a) if kind == "linear" else b * (b / a)
    if bound is not None and ((b > a and nxt > bound) or (b < a and nxt < bound)):
        return None
    return nxt


def select(levels: list, successes: list, n: list, clean_rate: float, targets: list) -> dict:
    """Apply the rule to one family's ladder (levels in ladder order, mildest first)."""
    if not 0 < clean_rate <= 1:
        raise ValueError("clean success rate must be in (0, 1]")
    rate = np.asarray(successes, float) / np.asarray(n, float)
    fit = isotonic_nonincreasing(rate, n)
    rel = fit / clean_rate
    chosen = {}
    for r in targets:
        d = np.abs(rel - r)
        # first = mildest among ties; equality up to float rounding (0.8-0.6 != 0.6-0.4 in binary)
        chosen[r] = levels[int(np.flatnonzero(np.isclose(d, d.min(), rtol=0, atol=TIE_ATOL))[0])]
    # a target is unreachable if even the strongest level stays above it (interpretation logged 2026-11-06: the
    # rule's "benign, keep only the top level" applies when NO target is reachable, not when only the lowest is not)
    unreachable = sorted(r for r in targets if rel[-1] > r)
    keep = sorted({chosen[r] for r in targets if r not in unreachable}, key=list(levels).index)
    return {"levels": list(levels), "rate": rate.tolist(), "fit": fit.tolist(), "relative": rel.tolist(),
            "chosen": chosen, "top_above_lowest_target": bool(rel[-1] > min(targets)),
            "unreachable_targets": unreachable, "selected_levels": keep or [levels[-1]], "benign": not keep}
