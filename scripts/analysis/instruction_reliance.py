"""Instruction reliance: how much of OFT's LIBERO-Goal success depends on the instruction, and what it does without the right one.

Usage:  python scripts/analysis/instruction_reliance.py   ->  results/instruction_reliance/summary.{json,md}

Conditions, all on the same 500 (task, init state) episodes:
  own        - the correct instruction (clean baseline, replicate 0)
  other_task - task (i+5) mod 10's instruction in task i's scene (Li et al. 2610.19438 probe)
  empty      - empty instruction
Reported: success with Wilson 95% CIs; language share (own - wrong) / own with a stratified cluster bootstrap CI
(tasks fixed, init states resampled; protocol statistics.bootstrap_*); exact McNemar vs own; per-task table; and,
from the cross-task goal oracle, which goals the policy achieved: the scene's own task, the PROMPTED task (it
followed the wrong instruction), another task, or none. Every run is ingested with completeness checks; a
completed-study verdict is refused for incomplete or dirty runs (vlamech.analysis.ingest).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.ingest import find_run, grid, load_run, paired  # noqa: E402
from vlamech.config import protocol  # noqa: E402
from vlamech.metrics.stats import hierarchical_bootstrap, mcnemar_exact, wilson_ci  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

LI_ET_AL = {"other_task": (0, 250)}  # OFT LIBERO-Goal wrong-instruction successes (Li et al. Tab. 17)


def goal_category(rec: dict, task: int, prompted: int) -> str:
    goals = {int(k) for k in rec["goal_first_step"]}
    if task in goals:
        return "own task goal"
    if prompted is not None and prompted in goals:
        return "PROMPTED task goal"
    if goals:
        return "other task goal"
    return "no goal"


def main():
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    n_tasks, n_states = suite.n_tasks, int(protocol("splits.n_init_states"))
    keys = grid(range(n_tasks), range(n_states))
    runs = {
        "own": load_run(find_run(ROOT / "results/baseline", "identity-*__ins-own__rep0"), keys),
        "other_task": load_run(find_run(ROOT / "results/instruction_reliance", "identity-*__ins-other_task__rep0"), keys),
        "empty": load_run(find_run(ROOT / "results/instruction_reliance", "identity-*__ins-empty__rep0"), keys),
    }
    for r in runs.values():
        r.require_complete()
    pairs = paired(runs["own"], runs["other_task"])
    paired(runs["own"], runs["empty"])

    rows = []
    for cond, rd in runs.items():
        for (t, s) in pairs:
            rec = rd.records[(t, s)]
            prompted = (t + 5) % n_tasks if cond == "other_task" else (t if cond == "own" else None)
            rows.append({"condition": cond, "task": t, "init_state": s, "success": bool(rec["success"]),
                         "category": goal_category(rec, t, prompted if cond == "other_task" else None)})
    df = pd.DataFrame(rows)
    out = {"accounting": {c: r.accounting() for c, r in runs.items()}, "conditions": {}}
    lines = ["# Instruction reliance (OpenVLA-OFT, LIBERO-Goal, 500 paired episodes per condition)", ""]
    lines += ["| Condition | Success | 95% CI | vs own: McNemar p |", "|---|---|---|---|"]
    own = df[df.condition == "own"].set_index(["task", "init_state"]).success
    for cond in runs:
        d = df[df.condition == cond].set_index(["task", "init_state"]).success
        k, n = int(d.sum()), len(d)
        lo, hi = wilson_ci(k, n)
        mc = mcnemar_exact(own.loc[d.index].values, d.values) if cond != "own" else None
        out["conditions"][cond] = {"success": k, "n": n, "rate": k / n, "wilson95": [lo, hi],
                                   "mcnemar_vs_own": mc}
        lines.append(f"| {cond} | {k}/{n} = {k/n:.1%} | [{lo:.1%}, {hi:.1%}] | "
                     f"{'—' if mc is None else f'{mc[chr(112)]:.2g}'} |")

    def share(sub):
        o = sub[sub.condition == "own"].success.mean()
        w = sub[sub.condition == "other_task"].success.mean()
        return (o - w) / o if o > 0 else np.nan

    sh = hierarchical_bootstrap(df[df.condition.isin(["own", "other_task"])], share,
                                levels=tuple(protocol("statistics.bootstrap_levels")),
                                strata=tuple(protocol("statistics.bootstrap_strata")),
                                n_boot=int(protocol("statistics.n_boot")))
    out["language_share"] = sh
    k0, n0 = LI_ET_AL["other_task"]
    lines += ["", f"Language share (own − wrong)/own = {sh['estimate']:.1%}  [95% CI {sh['ci_low']:.1%}, "
              f"{sh['ci_high']:.1%}]  (stratified bootstrap, {sh['n_boot']} replicates).",
              f"Li et al. 2610.19438 report wrong-instruction success {k0}/{n0} for OFT on LIBERO-Goal.", ""]

    lines += ["## What the policy achieved (cross-task goal oracle)", "",
              "| Condition | own task goal | PROMPTED (wrong) task goal | other task goal | no goal |", "|---|---|---|---|---|"]
    cats = ["own task goal", "PROMPTED task goal", "other task goal", "no goal"]
    out["categories"] = {}
    for cond in runs:
        c = df[df.condition == cond].category.value_counts()
        out["categories"][cond] = {k: int(c.get(k, 0)) for k in cats}
        lines.append(f"| {cond} | " + " | ".join(str(int(c.get(k, 0))) for k in cats) + " |")
    lines += ["", "(own: the PROMPTED column is empty by definition; for other_task, 'PROMPTED' = the policy "
              "achieved task (i+5)'s goal in task i's scene, i.e. it followed the wrong instruction.)", ""]

    lines += ["## Per task", "", "| Task | Instruction | own | other_task (prompted task) | empty |", "|---|---|---|---|---|"]
    per = df.groupby(["task", "condition"]).success.sum().unstack()
    out["per_task"] = {int(t): {c: int(per.loc[t, c]) for c in runs} for t in per.index}
    for t in range(n_tasks):
        lines.append(f"| {t} | {suite.get_task(t).language} | {per.loc[t, 'own']}/50 | {per.loc[t, 'other_task']}/50 "
                     f"(task {(t + 5) % n_tasks}) | {per.loc[t, 'empty']}/50 |")
    lines += ["", "## Empty instruction: which goal did the policy pursue by default?", ""]
    for t in range(n_tasks):
        recs = [runs["empty"].records[(t, s)] for s in range(n_states)]
        got = {}
        for r in recs:
            for g in r["goal_first_step"]:
                got[int(g)] = got.get(int(g), 0) + 1
        if got:
            desc = ", ".join(f"task {g} ({suite.get_task(g).language}) ×{n}" for g, n in sorted(got.items()))
            lines.append(f"- task {t} scene: {desc}")
    out["provenance_note"] = "runs may span several sessions; every session's commit is listed in accounting"
    atomic_write_json(ROOT / "results/instruction_reliance/summary.json", out)
    atomic_write_bytes(ROOT / "results/instruction_reliance/summary.md", ("\n".join(lines) + "\n").encode())
    print("\n".join(lines))
    print("\naccounting:", {c: (a["completed"], a["commits"], a["dirty_sessions"]) for c, a in out["accounting"].items()})


if __name__ == "__main__":
    main()
