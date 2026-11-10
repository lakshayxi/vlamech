"""Cross-model check: is the success lost under corruption specific to OFT? (protocol cross_model.*)

Usage (OFT environment):  python scripts/analysis/cross_model.py [--data-root DIR]  -> results/cross_model/summary.{json,md},
        results/figures/cross_model/success_by_model.png

pi0.5 runs (experiment cross_model) against the plain OFT runs of the corruption sweep, same task, init state and condition. Within pi0.5: exact
McNemar of each corruption against its own clean run, Holm over corruptions. Between models: difference of success drops
(pi0.5 drop minus OFT drop) with the stratified cluster bootstrap; descriptive, since the checkpoints differ.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.arms import load_outcomes, thresholds  # noqa: E402
from vlamech.config import protocol, split_states  # noqa: E402
from vlamech.metrics.stats import hierarchical_bootstrap, holm, mcnemar_exact, wilson_ci  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

MUTED, GRID, SURF, BLUE, ORANGE = "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"


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

    root, states, thr = a.data_root, split_states("confirmatory"), thresholds(a.data_root)
    conds = list(protocol("cross_model.conditions"))
    pi = {c: load_outcomes(root, "cross_model", c, "pi05", states, thr)[1] for c in conds}
    oft = {c: load_outcomes(root, "corruption_sweep", c, None, states, thr)[1] for c in conds}
    corrupted = [c for c in conds if c != "identity"]
    k = ["task", "init_state"]
    out, ps = {}, []
    for c in conds:
        n = len(pi[c])
        out[c] = {"pi05": float(pi[c].success.mean()), "pi05_ci": list(wilson_ci(int(pi[c].success.sum()), n)),
                  "oft": float(oft[c].success.mean()), "oft_ci": list(wilson_ci(int(oft[c].success.sum()), n)), "n": n}
    for c in corrupted:
        m = pi["identity"][k + ["success"]].merge(pi[c][k + ["success"]], on=k, suffixes=("_clean", "_c"))
        t = mcnemar_exact(m.success_clean.to_numpy(), m.success_c.to_numpy())
        ps.append(t["p"])
        d = pi["identity"][k + ["success"]].rename(columns={"success": "p0"}).merge(pi[c][k + ["success"]].rename(columns={"success": "p1"}), on=k)
        d = d.merge(oft["identity"][k + ["success"]].rename(columns={"success": "o0"}), on=k).merge(oft[c][k + ["success"]].rename(columns={"success": "o1"}), on=k)
        for col in ("p0", "p1", "o0", "o1"):
            d[col] = d[col].astype(float)
        out[c] |= {"drop_pi05": boot(d, lambda x: (x.p1 - x.p0).mean()), "drop_oft": boot(d, lambda x: (x.o1 - x.o0).mean()),
                   "drop_difference": boot(d, lambda x: ((x.p1 - x.p0) - (x.o1 - x.o0)).mean()), "p": t["p"], "n10": t["n10"], "n01": t["n01"]}
    for c, p in zip(corrupted, holm(ps)):
        out[c]["p_holm"] = float(p)
    f = lambda v: f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    L = ["# Cross-model check: pi0.5 against OFT (held-out, 300 episodes per cell)", "",
         "| condition | pi0.5 success | OFT success | pi0.5 change (pts) | OFT change (pts) | pi0.5 - OFT (pts) | p_holm (pi0.5) |", "|---|---|---|---|---|---|---|"]
    for c in conds:
        x = out[c]
        if c == "identity":
            L.append(f"| clean | {100 * x['pi05']:.1f}% | {100 * x['oft']:.1f}% | - | - | - | - |")
        else:
            L.append(f"| {c} | {100 * x['pi05']:.1f}% | {100 * x['oft']:.1f}% | {f(x['drop_pi05'])} | {f(x['drop_oft'])} | {f(x['drop_difference'])} | {x['p_holm']:.3g} |")
    base = root / "results" / "cross_model"
    base.mkdir(parents=True, exist_ok=True)
    atomic_write_json(base / "summary.json", out)
    atomic_write_bytes(base / "summary.md", ("\n".join(L) + "\n").encode())
    figs = root / "results" / "figures" / "cross_model"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for i, c in enumerate(conds):
        x = out[c]
        for key, col, mk, lab, off in (("oft", BLUE, "s", "OpenVLA-OFT", 0.1), ("pi05", ORANGE, "o", "π0.5", -0.1)):
            lo, hi = x[f"{key}_ci"]
            ax.plot([lo, hi], [i + off, i + off], color=col, lw=2, zorder=1)
            ax.scatter(x[key], i + off, s=55, color=col, marker=mk, zorder=2, label=lab if i == 0 else None)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(["clean" if c == "identity" else c for c in conds], fontsize=8)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("success rate (300 held-out episodes, 95% CI)", fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(figs / "success_by_model.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
