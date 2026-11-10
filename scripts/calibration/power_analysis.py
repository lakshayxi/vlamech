"""Power of the paired success-rate tests for the confirmatory split size.

Usage:  python scripts/calibration/power_analysis.py   ->  results/power_analysis/power.json + console table

Question answered: with n paired episodes per condition (confirmatory split: 30 init states x 10 tasks = 300),
what paired risk differences can the exact McNemar test detect at alpha = 0.05 (two-sided) with 80% power,
after Holm correction over m simultaneous contrasts?

Model: each episode pair is (clean outcome, treated outcome). Discordant cells p10 (clean success,
treated failure) and p01 (the reverse). For a given base success p0 and true difference d we use the
minimal-discordance configuration p01 = max(0, -d), p10 = max(0, d) plus a symmetric "noise" discordance
`q` (episodes that flip both ways for reasons unrelated to the treatment). OFT is not bitwise reproducible
under GPU load: Li et al. 2610.19438 measured 4% replicate discordance for OFT on 500 identical initial
states, so q = 0.04 is the reference value; the sweep brackets it, and the clean baseline measures q for our own setup.
Power is estimated by Monte Carlo (20,000 simulated studies per cell, fixed seed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vlamech.utils.io import atomic_write_json  # noqa: E402


def mcnemar_power(n: int, d: float, q: float, alpha: float, n_sim: int = 20_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    p10 = max(d, 0.0) + q / 2
    p01 = max(-d, 0.0) + q / 2
    if p10 + p01 > 1:
        return np.nan
    cells = rng.multinomial(n, [p10, p01, 1 - p10 - p01], size=n_sim)
    b, c = cells[:, 0], cells[:, 1]
    k = np.minimum(b, c)
    m = b + c
    # exact two-sided binomial p-value, vectorised via the CDF
    p = np.where(m > 0, np.minimum(1.0, 2 * stats.binom.cdf(k, np.maximum(m, 1), 0.5)), 1.0)
    return float((p < alpha).mean())


def main():
    alpha = 0.05
    ns = [100, 200, 300, 500]
    ds = [0.05, 0.08, 0.10, 0.15, 0.20]
    qs = [0.02, 0.04, 0.08]  # background discordance; 0.04 = Li et al.'s measured OFT replicate rate
    m_contrasts = [1, 10, 50]  # Holm: the smallest p-value is compared to alpha/m
    out = []
    for m in m_contrasts:
        for q in qs:
            for n in ns:
                for d in ds:
                    out.append({"m": m, "q": q, "n": n, "d": d, "power": mcnemar_power(n, d, q, alpha / m)})
    path = ROOT / "results" / "power_analysis" / "power.json"
    atomic_write_json(path, {"alpha": alpha, "test": "exact McNemar, Holm worst case alpha/m", "rows": out})
    print(f"wrote {path}\n")
    for m in m_contrasts:
        print(f"Holm-adjusted alpha = 0.05/{m}")
        for q in qs:
            print(f"  background discordance q={q}")
            print("    n   " + "  ".join(f"d={d:.2f}" for d in ds))
            for n in ns:
                row = [r["power"] for r in out if r["m"] == m and r["q"] == q and r["n"] == n]
                print(f"    {n:<4}" + "  ".join(f"{p:6.2f}" for p in row))


if __name__ == "__main__":
    main()
