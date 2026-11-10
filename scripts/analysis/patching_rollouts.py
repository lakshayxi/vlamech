"""Patching at states reached by the corrupted rollouts: analysis of scripts/run/patching_rollouts.py (protocol
patching.rollout_*).

Usage:  python scripts/analysis/patching_rollouts.py [--data-root DIR]  -> results/patching_rollouts/summary.{json,md},
        results/figures/patching_rollouts/steering_fail_vs_success.png

Same metric as the patching analysis (bowl IER_par against the clean render of the same state), per episode (mean over its states). On FAILING
rollouts, per condition (patching.rollout_tests; Holm over conditions x tests): (a) IER_par with the instruction clamp exceeds IER_par
with the other episode's clamp (absolute steering, so no division by a small loss); (b) injected corrupted states give IER_par
below 1; (c) lower than the other episode's clean states. rho is reported as a ratio of episode means. Failing against
succeeding rollouts: the difference in episode means of corrupted IER_par and of steering restored (stratified cluster
bootstrap, tasks fixed), descriptive; reported as undefined where a resample has no episode of a group.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.config import protocol  # noqa: E402
from vlamech.metrics import open_loop as ol  # noqa: E402
from vlamech.metrics.stats import hierarchical_bootstrap, holm, paired_sign_flip_test  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

MUTED, GRID, SURF, BLUE, ORANGE = "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"
TESTS = ("instr_minus_donor", "noising_minus_1", "noising_minus_donor")


def ids(name: str) -> tuple:
    t, rest = name[:-4].split("-init")
    i, q = rest.split("-q")
    return int(t[4:]), int(i), int(q)


def load(d: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(d.glob("task*-q*.npz")):
        z = np.load(f, allow_pickle=False)
        t, i, q = ids(f.name)
        clean, corr = z["clean"], z["corr"]
        p = float(ol.ier_par_against(corr, clean))
        rows.append({"task": t, "init_state": i, "q": q, "success": bool(z["success"]), "par_corr": p,
                     "par_prefix": float(ol.ier_par_against(z["prefix"], clean)),
                     "par_donor": float(ol.ier_par_against(z["donor_clamp"], clean)),
                     "noising": float(ol.ier_par_against(z["noising"], clean)),
                     "noising_donor": float(ol.ier_par_against(z["noising_donor"], clean))})
    return pd.DataFrame(rows)


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

    base = a.data_root / "results" / "patching_rollouts"
    conds = list(protocol("patching.patch_conditions"))
    alpha = float(protocol("statistics.alpha"))
    out, ps, keys = {}, [], []
    f = lambda v: "undefined" if v is None else f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"  # noqa: E731
    for c in conds:
        d = load(base / c)
        e = d.groupby(["task", "init_state", "success"]).mean(numeric_only=True).reset_index()
        fail, ok = e[~e.success].copy(), e[e.success].copy()
        fail["instr_minus_donor"] = fail.par_prefix - fail.par_donor
        fail["restored"], ok["restored"] = fail.par_prefix - fail.par_corr, ok.par_prefix - ok.par_corr
        fail["noising_minus_1"] = fail.noising - 1.0
        fail["noising_minus_donor"] = fail.noising - fail.noising_donor
        cell = {"n_states": int(len(d)), "n_failing": int(len(fail)), "n_succeeding": int(len(ok)),
                "par_corr_failing": boot(fail, lambda x: x.par_corr.mean()),
                "rho": boot(fail, lambda x: (x.par_prefix.mean() - x.par_corr.mean()) / (1 - x.par_corr.mean())),
                "rho_donor": boot(fail, lambda x: (x.par_donor.mean() - x.par_corr.mean()) / (1 - x.par_corr.mean())),
                "noising": boot(fail, lambda x: x.noising.mean()), "noising_donor": boot(fail, lambda x: x.noising_donor.mean())}
        cell["par_corr_succeeding"] = boot(ok, lambda x: x.par_corr.mean()) if len(ok) >= 5 else None
        for t in TESTS:
            cell[t] = boot(fail, lambda x: x[t].mean())
            cell[f"p_{t}"] = float(paired_sign_flip_test(fail[t].to_numpy(), n_perm=int(protocol("statistics.n_perm"))))
            ps.append(cell[f"p_{t}"])
            keys.append((c, t))
        if len(ok) >= 5:
            e["restored"] = e.par_prefix - e.par_corr
            for col in ("par_corr", "restored"):
                def diff(x):
                    a1, b1 = x[~x.success][col], x[x.success][col]
                    return float(a1.mean() - b1.mean()) if len(a1) and len(b1) else float("nan")
                try:
                    cell[f"{col}_fail_minus_success"] = boot(e, diff)
                except FloatingPointError:
                    cell[f"{col}_fail_minus_success"] = None
        out[c] = cell
    for (c, t), p in zip(keys, holm(ps)):
        out[c][f"p_holm_{t}"] = float(p)
    for c in conds:
        x = out[c]
        x["verdict"] = "instruction route" if (x["instr_minus_donor"][1] > 0 and x["p_holm_instr_minus_donor"] < alpha and
                                               all(x[t][2] < 0 and x[f"p_holm_{t}"] < alpha for t in TESTS[1:])) else "not supported"
    L = ["# Mechanism at states reached by the corrupted rollouts", "",
         "States: offsets 0 and +1 from the first chunk the rollout's own gripper closes (held-out sweep rollouts, 3 bowl tasks). Failing rollouts are tested; "
         "failing against succeeding is descriptive.", "",
         "| condition | states (failing/succeeding episodes) | corrupted IER_par, failing | rho instruction | rho other episode | instr - other | injected: corrupted | injected: other | corrupted - other | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        x = out[c]
        L.append(f"| {c} | {x['n_states']} ({x['n_failing']}/{x['n_succeeding']}) | {f(x['par_corr_failing'])} | {f(x['rho'])} | {f(x['rho_donor'])} | "
                 f"{f(x['instr_minus_donor'])} | {f(x['noising'])} | {f(x['noising_donor'])} | {f(x['noising_minus_donor'])} | {x['verdict']} |")
    L += ["", "## Failing against succeeding rollouts (descriptive)", "", "| condition | corrupted IER_par, failing | succeeding | failing - succeeding | steering restored, failing - succeeding |", "|---|---|---|---|---|"]
    for c in conds:
        x = out[c]
        L.append(f"| {c} | {f(x['par_corr_failing'])} | {f(x['par_corr_succeeding'])} | {f(x.get('par_corr_fail_minus_success'))} | {f(x.get('restored_fail_minus_success'))} |")
    atomic_write_json(base / "summary.json", out)
    atomic_write_bytes(base / "summary.md", ("\n".join(L) + "\n").encode())
    figs = a.data_root / "results" / "figures" / "patching_rollouts"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, c in enumerate(conds):
        x = out[c]
        for key, col, mk, lab, off in (("par_corr_failing", ORANGE, "o", "failing rollouts", -0.1), ("par_corr_succeeding", BLUE, "s", "succeeding rollouts", 0.1)):
            if x[key] is None:
                continue
            e0, lo, hi = x[key]
            ax.plot([lo, hi], [i + off, i + off], color=col, lw=2, zorder=1)
            ax.scatter(e0, i + off, s=55, color=col, marker=mk, zorder=2, label=lab if i == 0 else None)
    ax.axvline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds, fontsize=8)
    ax.set_xlabel("bowl steering at the rollout's grasp (IER_par, 1 = clean render of that state; 95% CI)", fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(figs / "steering_fail_vs_success.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
