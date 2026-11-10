"""Instruction effect under corruption on replayed clean states (H1), as pre-registered.

Usage:  python scripts/analyze_instruction_effect.py --pass <results/instruction_effect/<source run>> [--exploratory]
  -> <pass dir>/{summary.json, summary.md}

Per condition (family x severity): episode values from vlamech.metrics.open_loop (protocol metrics.ier_aggregation,
metrics.share_reference); cell estimate = mean episode log IER (exp = geometric-mean ratio) with the stratified
cluster bootstrap CI (tasks fixed, init states resampled) and the task-resampling sensitivity CI; test H0: mean log
IER = 0 by the episode-level sign-flip test; Holm across all cells. Verdict per cell:
"fade" (CI < 0 and significant), "reallocation" (CI > 0 and significant), else "no detectable change". VE and the
share difference are reported with CIs. A pooled summary across cells is descriptive only. --exploratory stamps the
output as exploratory (pipeline development on exploratory states).

Added 2026-11-07, before any held-out output was opened:
- H1-dir (secondary; protocol metrics.ier_decomposition): IER_par, IER_perp and cos_align per cell on the linear
  scale, with the same CIs; test H0: mean IER_par = 1 by the episode-level sign-flip test on IER_par - 1, Holm across
  cells; verdict "aligned fade" (CI < 1, significant), "aligned amplification" (CI > 1, significant), else "no
  detectable change". IER_perp and cos_align are descriptive.
- H1-task (secondary): do the bowl-placement tasks (protocol statistics.bowl_tasks) have a lower IER than the other
  tasks? Episode value = mean log IER over all corrupted cells; task value = mean over its episodes; exact one-sided
  task-relabelling test (vlamech.metrics.stats.task_group_permutation_test) and a stratified bootstrap CI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.instruction_effect import load  # noqa: E402
from vlamech.config import protocol  # noqa: E402
from vlamech.metrics.stats import (hierarchical_bootstrap, holm, paired_sign_flip_test,  # noqa: E402
                                   task_group_permutation_test)
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

def boot(df, statistic, sensitivity=False):
    lv = protocol("statistics.bootstrap_sensitivity_levels" if sensitivity else "statistics.bootstrap_levels")
    st = () if sensitivity else protocol("statistics.bootstrap_strata")
    r = hierarchical_bootstrap(df, statistic, levels=tuple(lv), strata=tuple(st),
                               n_boot=int(protocol("statistics.n_boot")), alpha=float(protocol("statistics.alpha")))
    return [r["estimate"], r["ci_low"], r["ci_high"]]


def ci(df, col, sensitivity=False):
    return boot(df, lambda d: d[col].mean(), sensitivity)


def group_difference(group):
    """T = mean over the group's tasks - mean over the other tasks, of task means of the episode values `v`."""
    def stat(d):
        t = d.groupby("task")["v"].mean()
        return t[t.index.isin(group)].mean() - t[~t.index.isin(group)].mean()
    return stat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="pass_dir", type=Path, required=True)
    ap.add_argument("--exploratory", action="store_true")
    a = ap.parse_args()
    df = load(a.pass_dir)
    cells = [c for c in dict.fromkeys(df["condition"]) if c != "identity"]
    alpha = float(protocol("statistics.alpha"))
    n_perm = int(protocol("statistics.n_perm"))
    out = {"exploratory": a.exploratory, "n_episodes": int(df[["task", "init_state"]].drop_duplicates().shape[0]),
           "cells": {}}
    ps, ps_par = [], []
    for c in cells:
        d = df[df.condition == c]
        p = paired_sign_flip_test(d["log_ier"].to_numpy(), n_perm=n_perm)
        p_par = paired_sign_flip_test(d["ier_par"].to_numpy() - 1.0, n_perm=n_perm)
        out["cells"][c] = {"log_ier": ci(d, "log_ier"), "log_ier_tasks_resampled": ci(d, "log_ier", True),
                           "ve": ci(d, "ve"), "share_diff": ci(d, "share_diff"), "p": p, "n": int(len(d)),
                           "ier_par": ci(d, "ier_par"), "ier_par_tasks_resampled": ci(d, "ier_par", True),
                           "ier_perp": ci(d, "ier_perp"), "cos_align": ci(d, "cos_align"), "p_par": p_par}
        ps.append(p)
        ps_par.append(p_par)
    for c, p, pp in zip(cells, holm(ps), holm(ps_par)):
        x = out["cells"][c]
        x["p_holm"], x["p_par_holm"] = float(p), float(pp)
        _, lo, hi = x["log_ier"]
        x["verdict"] = ("fade" if hi < 0 and p < alpha else "reallocation" if lo > 0 and p < alpha
                        else "no detectable change")
        _, lo, hi = x["ier_par"]
        x["verdict_par"] = ("aligned fade" if hi < 1 and pp < alpha else "aligned amplification" if lo > 1
                            and pp < alpha else "no detectable change")
    pooled = df[df.condition != "identity"]["log_ier"]
    out["pooled_descriptive"] = {"mean_log_ier": float(pooled.mean()), "n_cells": len(cells)}

    # H1-task (secondary)
    group = sorted(protocol("statistics.bowl_tasks"))
    ep = (df[df.condition != "identity"].groupby(["task", "init_state"])["log_ier"].mean()
          .rename("v").reset_index())
    tg = task_group_permutation_test(ep.groupby("task")["v"].mean().to_dict(), group)
    per_cell = {c: {"group": float(np.exp(df[(df.condition == c) & df.task.isin(group)].groupby("task")
                                          ["log_ier"].mean().mean())),
                    "rest": float(np.exp(df[(df.condition == c) & ~df.task.isin(group)].groupby("task")
                                         ["log_ier"].mean().mean()))} for c in cells}
    out["task_group"] = {"group": group, **tg, "T_ci": boot(ep, group_difference(group)),
                         "verdict": "lower" if tg["p"] < alpha else "no detectable difference",
                         "per_cell_descriptive": per_cell}
    atomic_write_json(a.pass_dir / "summary.json", out)

    L = [f"# Instruction effect under corruption{' (EXPLORATORY)' if a.exploratory else ''}", "",
         f"{out['n_episodes']} clean episodes, every captured state; IER = IE(corrupted) / IE(clean) on the same state. "
         "Intervals: stratified cluster bootstrap (tasks fixed); p: episode-level sign-flip, Holm across cells.", "",
         "| condition | IER (geometric mean) [95% CI] | tasks resampled | VE [CI] | share difference [CI] | p_holm | verdict |",
         "|---|---|---|---|---|---|---|"]
    for c in cells:
        x = out["cells"][c]
        e = [np.exp(v) for v in x["log_ier"]]
        s = [np.exp(v) for v in x["log_ier_tasks_resampled"]]
        L.append(f"| {c} | {e[0]:.2f} [{e[1]:.2f}, {e[2]:.2f}] | [{s[1]:.2f}, {s[2]:.2f}] | "
                 f"{x['ve'][0]:.2f} [{x['ve'][1]:.2f}, {x['ve'][2]:.2f}] | {x['share_diff'][0]:+.3f} "
                 f"[{x['share_diff'][1]:+.3f}, {x['share_diff'][2]:+.3f}] | {x['p_holm']:.2g} | {x['verdict']} |")
    L += ["", "## Direction of the instruction effect (H1-dir, secondary)", "",
          "Each instruction contrast split along the clean contrast (IER_par, 1 on clean) and across it (IER_perp, 0 on "
          "clean); cos_align = cosine to the clean contrast. Linear scale, arithmetic means; p: sign-flip on IER_par - 1, "
          "Holm across cells.", "",
          "| condition | IER_par [CI] | tasks resampled | IER_perp [CI] | cos_align [CI] | p_holm | verdict |",
          "|---|---|---|---|---|---|---|"]
    for c in cells:
        x = out["cells"][c]
        q, r, k, s = x["ier_par"], x["ier_perp"], x["cos_align"], x["ier_par_tasks_resampled"]
        L.append(f"| {c} | {q[0]:.3f} [{q[1]:.3f}, {q[2]:.3f}] | [{s[1]:.3f}, {s[2]:.3f}] | "
                 f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}] | {k[0]:.3f} [{k[1]:.3f}, {k[2]:.3f}] | "
                 f"{x['p_par_holm']:.2g} | {x['verdict_par']} |")
    g = out["task_group"]
    L += ["", f"## Bowl-placement tasks {g['group']} vs the other tasks (H1-task, secondary)", "",
          f"T = {g['T']:+.3f} [{g['T_ci'][1]:+.3f}, {g['T_ci'][2]:+.3f}] (difference of task-mean log IER over all "
          f"corrupted cells; ratio {np.exp(g['T']):.2f}); exact one-sided p = {g['p']:.3g} over {g['n_relabellings']} "
          f"task relabellings; verdict: {g['verdict']}.", "",
          "| condition | IER, group | IER, other tasks |", "|---|---|---|"]
    L += [f"| {c} | {v['group']:.2f} | {v['rest']:.2f} |" for c, v in g["per_cell_descriptive"].items()]
    atomic_write_bytes(a.pass_dir / "summary.md", ("\n".join(L) + "\n").encode())
    print("\n".join(L))


if __name__ == "__main__":
    main()
