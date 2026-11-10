"""Mitigation: arm selection on the exploratory split and the held-out test (protocol mitigation.*; H5).

Usage:  python scripts/analyze_mitigation.py --select    -> results/mitigation_selection/summary.{json,md}, configs/frozen_mitigation.yaml (once)
        python scripts/analyze_mitigation.py --confirm   -> results/mitigation_test/summary.{json,md}, results/figures/mitigation/*
        python scripts/analyze_mitigation.py --followup  -> results/mitigation_selection/followup.{json,md} (mitigation.followup_arms, exploratory)

Selection (mitigation.selection_rule): the arm with the highest mean success over the corrupted conditions, among arms
whose clean success is not significantly below the plain arm's (exact McNemar); ties go to the arm listed first.
Held-out test: each frozen arm against the plain corruption-sweep episodes (same task, init state, condition), exact McNemar, Holm
over arms x corrupted conditions; clean change reported separately with its CI; stratified cluster bootstrap CIs.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.arms import load_outcomes, thresholds  # noqa: E402
from vlamech.config import FROZEN_MITIGATION, protocol, split_states  # noqa: E402
from vlamech.metrics.stats import hierarchical_bootstrap, holm, mcnemar_exact  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

PLAIN = "cag:1"
FROZEN = FROZEN_MITIGATION
INK, MUTED, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"


def diff_ci(a: pd.DataFrame, b: pd.DataFrame, col: str) -> list:
    d = a[["task", "init_state", col]].merge(b[["task", "init_state", col]], on=["task", "init_state"], suffixes=("_a", "_b"))
    if len(d) != len(a) or len(d) != len(b):
        raise RuntimeError("runs are not paired")
    d["d"] = d[f"{col}_b"].astype(float) - d[f"{col}_a"].astype(float)
    r = hierarchical_bootstrap(d, lambda x: x["d"].mean(), levels=tuple(protocol("statistics.bootstrap_levels")),
                               strata=tuple(protocol("statistics.bootstrap_strata")),
                               n_boot=int(protocol("statistics.n_boot")), alpha=float(protocol("statistics.alpha")))
    return [r["estimate"], r["ci_low"], r["ci_high"]]


def test(a: pd.DataFrame, b: pd.DataFrame, col: str = "success") -> dict:
    m = a[["task", "init_state", col]].merge(b[["task", "init_state", col]], on=["task", "init_state"], suffixes=("_a", "_b"))
    return mcnemar_exact(m[f"{col}_a"].to_numpy(), m[f"{col}_b"].to_numpy())


def select():
    if FROZEN.exists():
        raise FileExistsError(f"{FROZEN} exists: the selection is written once")
    arms, conds = list(protocol("mitigation.arms")), list(protocol("mitigation.selection_conditions"))
    states, thr, alpha = split_states("exploratory"), thresholds(ROOT), float(protocol("statistics.alpha"))
    data = {(a, c): load_outcomes(ROOT, "mitigation_selection", c, a, states, thr)[1] for a in arms for c in conds}
    corrupted = [c for c in conds if c != "identity"]
    rows = {}
    for a in arms:
        t = test(data[(PLAIN, "identity")], data[(a, "identity")])
        harm = t["risk_diff"] < 0 and t["p"] < alpha
        rows[a] = {"clean": float(data[(a, "identity")].success.mean()), "clean_change": t["risk_diff"], "clean_p": t["p"],
                   "harm": bool(harm), "corrupted": float(np.mean([data[(a, c)].success.mean() for c in corrupted])),
                   "per_condition": {c: float(data[(a, c)].success.mean()) for c in conds}}
    eligible = [a for a in arms if not rows[a]["harm"]]
    selected = max(eligible, key=lambda a: (rows[a]["corrupted"], -arms.index(a)))
    frozen_arms = list(dict.fromkeys(x for x in (selected, "cag:3") if x != PLAIN))
    out = {"selected": selected, "frozen_arms": frozen_arms, "arms": rows}
    L = ["# Mitigation arm selection (exploratory, 150 episodes per cell)", "",
         f"Selected: **{selected}**. Held-out arms: {', '.join(frozen_arms)}.", "",
         "| arm | clean success | change vs plain (p) | " + " | ".join(corrupted) + " | mean corrupted |",
         "|---|---|---|" + "---|" * (len(corrupted) + 1)]
    for a in arms:
        r = rows[a]
        L.append(f"| {a} | {r['clean']:.3f} | {r['clean_change']:+.3f} ({r['clean_p']:.2g}){' harm' if r['harm'] else ''} | "
                 + " | ".join(f"{r['per_condition'][c]:.3f}" for c in corrupted) + f" | {r['corrupted']:.3f} |")
    (ROOT / "results" / "mitigation_selection").mkdir(parents=True, exist_ok=True)
    atomic_write_json(ROOT / "results" / "mitigation_selection" / "summary.json", out)
    atomic_write_bytes(ROOT / "results" / "mitigation_selection" / "summary.md", ("\n".join(L) + "\n").encode())
    FROZEN.write_text("# Mitigation arm frozen by the selection rule on exploratory data. Written once by\n"
                      "# scripts/analyze_mitigation.py --select, never edited.\n" + yaml.safe_dump(
        {"frozen": datetime.datetime.now().isoformat(timespec="seconds"), "selected": selected, "arms": frozen_arms}))
    print("\n".join(L))


def followup():
    conds, states, thr = list(protocol("mitigation.selection_conditions")), split_states("exploratory"), thresholds(ROOT)
    plain = {c: load_outcomes(ROOT, "mitigation_selection", c, PLAIN, states, thr)[1] for c in conds}
    arms = list(protocol("mitigation.followup_arms"))
    cells, L = {}, ["# Guidance follow-up (exploratory, 150 paired episodes per cell, outside the arm selection)", "",
                    "| arm | condition | plain | arm | change [CI] | discordant (lost/gained) | p | collisions plain / arm |",
                    "|---|---|---|---|---|---|---|---|"]
    ps, keys = [], []
    for a in arms:
        for c in conds:
            x = load_outcomes(ROOT, "mitigation_selection", c, a, states, thr)[1]
            t = test(plain[c], x)
            cells[f"{a}|{c}"] = {"plain": float(plain[c].success.mean()), "arm": float(x.success.mean()),
                                 "change": diff_ci(plain[c], x, "success"), "p": t["p"], "n10": t["n10"], "n01": t["n01"],
                                 "collision_plain": float(plain[c].collision.mean()), "collision_arm": float(x.collision.mean())}
            if c != "identity":
                ps.append(t["p"])
                keys.append(f"{a}|{c}")
    for k, p in zip(keys, holm(ps)):
        cells[k]["p_holm"] = float(p)
    for k, x in cells.items():
        a, c = k.split("|")
        p = f"{x['p_holm']:.3g} (Holm)" if "p_holm" in x else f"{x['p']:.3g}"
        L.append(f"| {a} | {c} | {x['plain']:.3f} | {x['arm']:.3f} | {x['change'][0]:+.3f} [{x['change'][1]:+.3f}, {x['change'][2]:+.3f}] "
                 f"| {x['n10']}/{x['n01']} | {p} | {x['collision_plain']:.3f} / {x['collision_arm']:.3f} |")
    atomic_write_json(ROOT / "results" / "mitigation_selection" / "followup.json", cells)
    atomic_write_bytes(ROOT / "results" / "mitigation_selection" / "followup.md", ("\n".join(L) + "\n").encode())
    print("\n".join(L))


def confirm():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = yaml.safe_load(FROZEN.read_text())["arms"]
    conds = list(protocol("mitigation.selection_conditions"))
    states, thr = split_states("confirmatory"), thresholds(ROOT)
    plain = {c: load_outcomes(ROOT, "corruption_sweep", c, None, states, thr)[1] for c in conds}
    arm = {(a, c): load_outcomes(ROOT, "mitigation_test", c, a, states, thr)[1] for a in arms for c in conds}
    cells, ps, keys = {}, [], []
    for a in arms:
        for c in conds:
            t = test(plain[c], arm[(a, c)])
            cells[f"{a}|{c}"] = {"plain": float(plain[c].success.mean()), "arm": float(arm[(a, c)].success.mean()),
                                 "change": diff_ci(plain[c], arm[(a, c)], "success"), "p": t["p"], "n10": t["n10"], "n01": t["n01"],
                                 "collision_plain": float(plain[c].collision.mean()), "collision_arm": float(arm[(a, c)].collision.mean()),
                                 "collision_change": diff_ci(plain[c], arm[(a, c)], "collision")}
            if c != "identity":
                ps.append(t["p"])
                keys.append(f"{a}|{c}")
    for k, p in zip(keys, holm(ps)):
        cells[k]["p_holm"] = float(p)
    out = {"arms": arms, "cells": cells}
    L = ["# Mitigation held-out test (300 paired episodes per cell, against the plain sweep episodes)", "",
         "| arm | condition | plain | arm | change [CI] | discordant (lost/gained) | p_holm | collisions plain / arm |",
         "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        for c in conds:
            x = cells[f"{a}|{c}"]
            p = f"{x['p_holm']:.3g}" if "p_holm" in x else f"{x['p']:.3g} (clean)"
            L.append(f"| {a} | {c} | {x['plain']:.3f} | {x['arm']:.3f} | {x['change'][0]:+.3f} [{x['change'][1]:+.3f}, {x['change'][2]:+.3f}] "
                     f"| {x['n10']}/{x['n01']} | {p} | {x['collision_plain']:.3f} / {x['collision_arm']:.3f} |")
    base = ROOT / "results" / "mitigation_test"
    base.mkdir(parents=True, exist_ok=True)
    atomic_write_json(base / "summary.json", out)
    atomic_write_bytes(base / "summary.md", ("\n".join(L) + "\n").encode())
    figs = ROOT / "results" / "figures" / "mitigation"
    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.4), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], arms):
        for i, c in enumerate(conds):
            x = cells[f"{a}|{c}"]
            ax.plot([x["plain"], x["arm"]], [i, i], color=GRID, lw=2, zorder=1)
            ax.scatter(x["plain"], i, s=60, color=MUTED, zorder=2, label="plain" if i == 0 else None)
            ax.scatter(x["arm"], i, s=60, color=BLUE, marker="s", zorder=2, label=a if i == 0 else None)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("success rate (300 episodes)", fontsize=8)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(a, fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(figs / "success.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--followup", action="store_true")
    a = ap.parse_args()
    if a.select + a.confirm + a.followup != 1:
        raise SystemExit("exactly one of --select / --confirm / --followup")
    select() if a.select else confirm() if a.confirm else followup()
