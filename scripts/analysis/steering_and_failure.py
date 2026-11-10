"""Does losing steering go with failing? Held-out open-loop measures against the closed-loop outcome of the same episode.

Usage:  python scripts/analysis/steering_and_failure.py [--data-root DIR]  -> results/steering_and_failure/{summary.json, summary.md},
        results/figures/steering_and_failure/steering_vs_success.png

Post hoc and not pre-registered (added 2026-11-09); correlational: the open-loop values come from clean states, the
outcomes from corrupted closed-loop runs of the same task, init state and condition. Corrupted cells only.
  Between cells: Spearman correlation of cell means (task x condition) with cell success, raw and with task and condition
  means removed.
  Within cells: failure AUROC of each measure after removing the mean of its (task, condition), i.e. among episodes of
  the same task under the same corruption.
Measures and the direction expected to go with failure: IER_par (lower), cos_align (lower), IER_perp (higher); log IER
(the size of the effect) has no predicted direction and is scored as higher. CIs: stratified cluster bootstrap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from vlamech.analysis.instruction_effect import load  # noqa: E402
from vlamech.analysis.sweep import build_table  # noqa: E402
from vlamech.config import protocol  # noqa: E402
from vlamech.metrics.stats import auroc, hierarchical_bootstrap  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

MEASURES = {"ier_par": -1.0, "cos_align": -1.0, "ier_perp": 1.0, "log_ier": 1.0}
INK, MUTED, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"


def cell_means(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby(["task", "condition"])[[*MEASURES, "success"]].mean().reset_index()


def two_way(c: pd.DataFrame, col: str) -> np.ndarray:
    return (c[col] - c.groupby("task")[col].transform("mean") - c.groupby("condition")[col].transform("mean") + c[col].mean()).to_numpy()


def spearman(c: pd.DataFrame, m: str, adjusted: bool) -> float:
    x, y = (two_way(c, m), two_way(c, "success")) if adjusted else (c[m].to_numpy(), c["success"].to_numpy())
    return float(stats.spearmanr(x, y).statistic)


def within_auroc(d: pd.DataFrame, m: str) -> float:
    res = d[m] - d.groupby(["task", "condition"])[m].transform("mean")
    return auroc(MEASURES[m] * res.to_numpy(), (~d["success"]).to_numpy())


def boot(d: pd.DataFrame, stat) -> list:
    r = hierarchical_bootstrap(d, stat, levels=tuple(protocol("statistics.bootstrap_levels")),
                               strata=tuple(protocol("statistics.bootstrap_strata")),
                               n_boot=int(protocol("statistics.n_boot")), alpha=float(protocol("statistics.alpha")))
    return [r["estimate"], r["ci_low"], r["ci_high"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=ROOT)
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ol = load(a.data_root / "results" / "instruction_effect" / "identity-agentview+wrist__ins-own__rep1")
    rows, _, skipped, _ = build_table(a.data_root, False)
    if skipped:
        raise RuntimeError(f"sweep conditions missing: {skipped}")
    sweep = pd.DataFrame(rows).rename(columns={"init": "init_state"})[["task", "init_state", "condition", "success", "outcome_v2"]]
    d = ol[ol.condition != "identity"].merge(sweep, on=["task", "init_state", "condition"], how="left", validate="one_to_one")
    if d["success"].isna().any() or len(d) != 300 * ol.condition.nunique() - 300:
        raise RuntimeError(f"open-loop and closed-loop episodes do not pair: {len(d)} rows")
    d["success"] = d["success"].astype(bool)

    out, L = {"n_episodes": int(len(d)), "n_cells": int(d.groupby(["task", "condition"]).ngroups)}, []
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"  # noqa: E731
    table = {}
    for m in MEASURES:
        table[m] = {"between_raw": boot(d, lambda x: spearman(cell_means(x), m, False)),
                    "between_adjusted": boot(d, lambda x: spearman(cell_means(x), m, True)),
                    "within_auroc": boot(d, lambda x: within_auroc(x, m))}
    out["measures"] = table
    names = {"ier_par": "along the clean direction (IER_par)", "cos_align": "alignment (cos_align)",
             "ier_perp": "across it (IER_perp)", "log_ier": "size (log IER)"}
    L += ["# Steering loss and closed-loop failure (post hoc, not pre-registered; correlational)", "",
          f"{out['n_episodes']} corrupted episodes in {out['n_cells']} task x condition cells; held-out open-loop values against the "
          "closed-loop outcome of the same episode.", "",
          "| measure | Spearman with cell success | same, task and condition removed | failure AUROC within a cell |", "|---|---|---|---|"]
    L += [f"| {names[m]} | {f(table[m]['between_raw'])} | {f(table[m]['between_adjusted'])} | {table[m]['within_auroc'][0]:.3f} [{table[m]['within_auroc'][1]:.3f}, {table[m]['within_auroc'][2]:.3f}] |" for m in MEASURES]
    base = a.data_root / "results" / "steering_and_failure"
    base.mkdir(parents=True, exist_ok=True)
    atomic_write_json(base / "summary.json", out)
    atomic_write_bytes(base / "summary.md", ("\n".join(L) + "\n").encode())

    c = cell_means(d)
    bowl = c.task.isin(protocol("statistics.bowl_tasks"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, m, xl in zip(axes, ("ier_par", "log_ier"), ("steering along the clean direction (IER_par)", "size of the instruction effect (log IER)")):
        ax.scatter(c.loc[~bowl, m], c.loc[~bowl, "success"], s=26, color=BLUE, marker="o", edgecolor=SURF, lw=0.5, label="other tasks")
        ax.scatter(c.loc[bowl, m], c.loc[bowl, "success"], s=30, color=ORANGE, marker="s", edgecolor=SURF, lw=0.5, label="bowl tasks")
        ax.set_xlabel(xl, fontsize=8)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"Spearman {table[m]['between_raw'][0]:+.2f}", fontsize=9, loc="left")
    axes[0].set_ylabel("closed-loop success (task x condition cell)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Steering measures against closed-loop success across task and condition cells; held-out data, post hoc", fontsize=10, x=0.01, ha="left", y=1.02)
    figs = a.data_root / "results" / "figures" / "steering_and_failure"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "steering_vs_success.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
