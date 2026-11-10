"""Held-out corruption sweep, analysed exactly as pre-registered (configs/protocol.yaml statistics).

Usage:  python scripts/analysis/corruption_sweep.py  ->  results/corruption_sweep/{summary.json, summary.md}, results/figures/corruption_sweep/*.png

Data: every frozen condition (configs/frozen_levels.yaml), strict ingestion, frozen oracle v2 (vlamech.analysis.sweep).
Pairing: every condition is paired with the clean run on (task, init state). Uncertainty: stratified cluster bootstrap
(tasks fixed strata, init states resampled; protocol statistics.*), with the task-resampling sensitivity interval
reported next to it. Tests and multiplicity families (Holm within each):
  success      corrupted vs clean, McNemar exact, every corrupted condition
  H2 rate      intent-loss rate (v2) corrupted vs clean, McNemar exact
  H2 share     intent-loss share among failures vs clean, Fisher exact
  trend        success on severity (log for geometric ladders, linear for the photometric 0.5-stop ladder; the
               ladder's own spacing, protocol corruption.ladder_extension), task fixed effects, cluster-robust SE on
               (task, init state); families with >= 2 levels
  controls     target_occlusion vs matched_bg_box; bg_patches vs its exact-model-input variant (McNemar exact)
Reported, not hypotheses: hazard proxies (collision above the clean baseline maximum, object off the table) with paired
differences; the exploratory destination-aware intent loss (protocol oracle.destination_label) with CIs, no tests.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.sweep import build_table  # noqa: E402
from vlamech.config import protocol  # noqa: E402
from vlamech.metrics.stats import hierarchical_bootstrap, holm, mcnemar_exact, wilson_ci  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

OUT = ROOT / "results" / "corruption_sweep"
FIG = ROOT / "results" / "figures" / "corruption_sweep"
INK, MUTED, GRID, SURF, BLUE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6"
SEVERITY = {"blur": ("log", lambda v: math.log(v)), "noise": ("log", lambda v: math.log(v)),
            "jpeg": ("log", lambda v: -math.log(v)),  # severity rises as quality falls
            "photometric": ("linear", lambda v: v)}


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables no more likely than the observed one)."""
    n1, n2, k = a + b, c + d, a + c
    def h(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    p0 = h(a)
    return float(min(1.0, sum(h(x) for x in range(max(0, k - n2), min(n1, k) + 1) if h(x) <= p0 * (1 + 1e-9))))


def boot(df, stat, sensitivity=False):
    levels = protocol("statistics.bootstrap_sensitivity_levels") if sensitivity else protocol("statistics.bootstrap_levels")
    strata = () if sensitivity else protocol("statistics.bootstrap_strata")
    r = hierarchical_bootstrap(df, stat, levels=tuple(levels), strata=tuple(strata),
                               n_boot=int(protocol("statistics.n_boot")), seed=0, alpha=float(protocol("statistics.alpha")))
    return [r["estimate"], r["ci_low"], r["ci_high"]]


def trend(df_fam: pd.DataFrame) -> dict:
    import statsmodels.api as sm

    X = pd.get_dummies(df_fam["task"].astype(str), prefix="task", drop_first=True, dtype=float)
    X.insert(0, "severity", df_fam["severity"].astype(float))
    X = sm.add_constant(X)
    groups = pd.factorize(df_fam["task"].astype(str) + "|" + df_fam["init_state"].astype(str))[0]
    fit = sm.GLM(df_fam["success"].astype(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    lo, hi = fit.conf_int(alpha=float(protocol("statistics.alpha"))).loc["severity"]
    return {"slope": float(fit.params["severity"]), "ci": [float(lo), float(hi)], "p": float(fit.pvalues["severity"]),
            "n": int(len(df_fam)), "converged": bool(fit.converged)}


def main():
    rows, _, _, _ = build_table(ROOT, partial=False)
    ep = pd.DataFrame(rows).rename(columns={"init": "init_state"})
    ep["intent_v2"] = ep["outcome_v2"] == "intent_loss"
    ep["intent_dest"] = ep["outcome_dest"].isin(["intent_loss", "wrong_destination"])
    conds = list(dict.fromkeys(ep["condition"]))
    corrupted = [c for c in conds if c != "identity"]
    wide = ep.pivot(index=["task", "init_state"], columns="condition",
                    values=["success", "intent_v2", "intent_dest", "collision", "off_table"]).reset_index()
    wide.columns = [a if not b else f"{a}|{b}" for a, b in wide.columns]
    if wide.isna().any().any():
        raise RuntimeError("unpaired episodes: every condition must cover every (task, init state)")

    def col(m, c):
        return wide[f"{m}|{c}"].to_numpy(bool)

    out = {"n_pairs": int(len(wide)), "conditions": {}, "families": {}}
    for c in conds:
        k, n = int(col("success", c).sum()), len(wide)
        fails = ~col("success", c)
        out["conditions"][c] = {
            "success": k, "n": n, "rate_wilson95": list(wilson_ci(k, n)),
            "rate_boot": boot(wide, lambda d, c=c: d[f"success|{c}"].mean()),
            "intent_v2": int(col("intent_v2", c).sum()), "intent_dest": int(col("intent_dest", c).sum()),
            "failures": int(fails.sum()), "intent_v2_among_failures": int((col("intent_v2", c) & fails).sum()),
            "collision": int(col("collision", c).sum()), "off_table": int(col("off_table", c).sum())}

    fams = {"success": "success", "h2_rate": "intent_v2", "collision": "collision", "off_table": "off_table"}
    for fam, m in fams.items():
        res, ps = {}, []
        for c in corrupted:
            t = mcnemar_exact(col(m, "identity"), col(m, c))
            rd = boot(wide, lambda d, c=c, m=m: d[f"{m}|{c}"].mean() - d[f"{m}|identity"].mean())
            res[c] = {"mcnemar": t, "rd_boot": rd}
            if fam == "success":
                res[c]["rd_boot_tasks_resampled"] = boot(
                    wide, lambda d, c=c: d[f"success|{c}"].mean() - d["success|identity"].mean(), sensitivity=True)
            ps.append(t["p"])
        for c, p in zip(corrupted, holm(ps)):
            res[c]["p_holm"] = float(p)
        out["families"][fam] = res

    clean = out["conditions"]["identity"]
    res, ps = {}, []
    for c in corrupted:
        x = out["conditions"][c]
        a, b = x["intent_v2_among_failures"], x["failures"] - x["intent_v2_among_failures"]
        cc, d = clean["intent_v2_among_failures"], clean["failures"] - clean["intent_v2_among_failures"]
        p = fisher_exact(a, b, cc, d)
        res[c] = {"share": a / x["failures"] if x["failures"] else None, "clean_share": cc / clean["failures"],
                  "table": [[a, b], [cc, d]], "p": p}
        ps.append(p)
    for c, p in zip(corrupted, holm(ps)):
        res[c]["p_holm"] = float(p)
    out["families"]["h2_share"] = res

    res = {}
    for c in corrupted:
        res[c] = {"rd_boot": boot(wide, lambda d, c=c: d[f"intent_dest|{c}"].mean() - d["intent_dest|identity"].mean())}
    out["exploratory_destination_aware_intent"] = res

    res, ps, names = {}, [], []
    for fam, (kind, f) in SEVERITY.items():
        levels = [c for c in corrupted if c.split("-")[0] == fam and "exactmi" not in c]
        if len(levels) < 2:
            continue
        sub = ep[ep["condition"].isin(levels)].copy()
        sub["severity"] = [f(float(c.split("-")[1])) for c in sub["condition"]]
        res[fam] = trend(sub) | {"scale": kind, "levels": levels}
        ps.append(res[fam]["p"]), names.append(fam)
    for fam, p in zip(names, holm(ps)):
        res[fam]["p_holm"] = float(p)
    out["families"]["trend"] = res

    res, ps = {}, []
    exactmi = [c for c in corrupted if c.endswith("exactmi")]
    pairs = [("target_occlusion", "matched_bg_box")] + [(c.replace("-exactmi", ""), c) for c in exactmi]
    pairs = [p for p in pairs if set(p) <= set(conds)]  # build_table(partial=False) guarantees every condition
    for a, b in pairs:
        t = mcnemar_exact(col("success", b), col("success", a))
        res[f"{a} vs {b}"] = {"mcnemar": t, "rd_boot": boot(wide, lambda d, a=a, b=b: d[f"success|{a}"].mean()
                                                             - d[f"success|{b}"].mean())}
        ps.append(t["p"])
    for k, p in zip(res, holm(ps) if ps else []):
        res[k]["p_holm"] = float(p)
    out["families"]["controls"] = res

    atomic_write_json(OUT / "summary.json", out)
    md = report(out, conds, corrupted)
    atomic_write_bytes(OUT / "summary.md", md.encode())
    figures(out, conds, ep)
    print(md)


def _ci(x, pct=True):
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.2f}")
    return f"{f(x[0])} [{f(x[1])}, {f(x[2])}]"


def report(out, conds, corrupted) -> str:
    F = out["families"]
    L = ["# Corruption sweep (held-out)", "",
         f"{out['n_pairs']} (task, init state) pairs per condition, held-out confirmatory split, frozen selection and "
         "oracle v2. Intervals: stratified cluster bootstrap (tasks fixed, init states resampled), percentile 95%; "
         "'tasks resampled' = sensitivity interval. p_holm: Holm within the family named in the header.", "",
         "## Success (family: success)", "",
         "| condition | success | rate % [95% CI] | difference vs clean, pts [CI] | tasks resampled | McNemar n10/n01 | p_holm |",
         "|---|---|---|---|---|---|---|"]
    for c in conds:
        x = out["conditions"][c]
        if c == "identity":
            L.append(f"| {c} | {x['success']}/{x['n']} | {_ci(x['rate_boot'])} | - | - | - | - |")
            continue
        s = F["success"][c]
        L.append(f"| {c} | {x['success']}/{x['n']} | {_ci(x['rate_boot'])} | {_ci(s['rd_boot'])} | "
                 f"{_ci(s['rd_boot_tasks_resampled'])} | {s['mcnemar']['n10']}/{s['mcnemar']['n01']} | {s['p_holm']:.2g} |")
    L += ["", "## H2: intent loss (oracle v2)", "",
          "| condition | intent-loss episodes | rate difference vs clean, pts [CI] | p_holm (rate) | share among failures "
          "(clean) | p_holm (share, Fisher) | destination-aware difference, pts [CI] (exploratory) |",
          "|---|---|---|---|---|---|---|"]
    for c in corrupted:
        r, s, e = F["h2_rate"][c], F["h2_share"][c], out["exploratory_destination_aware_intent"][c]
        share = "-" if s["share"] is None else f"{100 * s['share']:.0f}%"
        L.append(f"| {c} | {out['conditions'][c]['intent_v2']} | {_ci(r['rd_boot'])} | {r['p_holm']:.2g} | {share} "
                 f"({100 * s['clean_share']:.0f}%) | {s['p_holm']:.2g} | {_ci(e['rd_boot'])} |")
    L += ["", "## Severity trends (family: trend)", "", "| family | scale | levels | slope (log-odds per unit) [CI] | p_holm |",
          "|---|---|---|---|---|"]
    for fam, t in F["trend"].items():
        L.append(f"| {fam} | {t['scale']} | {', '.join(t['levels'])} | {t['slope']:.2f} [{t['ci'][0]:.2f}, "
                 f"{t['ci'][1]:.2f}] | {t['p_holm']:.2g} |")
    L += ["", "## Controls (family: controls)", "", "| contrast | success difference, pts [CI] | McNemar n10/n01 | p_holm |",
          "|---|---|---|---|"]
    for k, t in F["controls"].items():
        L.append(f"| {k} | {_ci(t['rd_boot'])} | {t['mcnemar']['n10']}/{t['mcnemar']['n01']} | {t['p_holm']:.2g} |")
    L += ["", "## Hazard proxies (reported; Holm within each proxy)", "",
          "| condition | collision > clean max | difference, pts [CI] | p_holm | off table | p_holm |", "|---|---|---|---|---|---|"]
    for c in corrupted:
        a, b = F["collision"][c], F["off_table"][c]
        L.append(f"| {c} | {out['conditions'][c]['collision']} | {_ci(a['rd_boot'])} | {a['p_holm']:.2g} | "
                 f"{out['conditions'][c]['off_table']} | {b['p_holm']:.2g} |")
    L.append(f"\nClean: {out['conditions']['identity']['collision']} collisions above the clean-success maximum (the "
             "proxy's false-alarm floor), off table " f"{out['conditions']['identity']['off_table']}.")
    return "\n".join(L) + "\n"


def figures(out, conds, ep):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    order = ["identity"] + [c for c in conds if c != "identity"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURF, gridspec_kw={"width_ratios": [1, 1.3]})
    y = np.arange(len(order))[::-1]
    for yi, c in zip(y, order):
        e, lo, hi = out["conditions"][c]["rate_boot"]
        ax[0].plot([lo, hi], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
        ax[0].plot(e, yi, "o", color=BLUE if c != "identity" else INK, ms=7, mec=SURF, mew=1.5)
    clean = out["conditions"]["identity"]["rate_boot"][0]
    ax[0].axvline(clean, color=MUTED, lw=1, ls="--")
    ax[0].set_yticks(y, order, fontsize=8)
    ax[0].set_xlim(0, 1.02)
    ax[0].set_xlabel("task success (95% CI, stratified cluster bootstrap)", fontsize=9, color=INK)
    cats = [("success", BLUE, "success"), ("intent_loss", "#eb6834", "wrong task (v2)"),
            ("execution_loss", "#1baf7a", "right task, botched (v2)"), ("no_progress", MUTED, "stalled")]
    left = np.zeros(len(order))
    for key, colr, lab in cats:
        vals = np.array([(ep[(ep.condition == c)]["outcome_v2"] == key).mean() for c in order])
        ax[1].barh(y, vals, left=left, color=colr, label=lab, height=0.72, edgecolor=SURF, linewidth=1)
        left += vals
    wd = [int(((ep.condition == c) & (ep.outcome_dest == "wrong_destination")).sum()) for c in order]
    for yi, w in zip(y, wd):
        if w > 0:
            ax[1].text(1.01, yi, f"{w} wrong destination", va="center", fontsize=7, color=MUTED)
    ax[1].set_yticks(y, [""] * len(order))
    ax[1].set_xlim(0, 1.25)
    ax[1].set_xlabel("share of episodes by outcome (frozen oracle v2); right: exploratory destination label",
                     fontsize=9, color=INK)
    ax[1].legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Corruption sweep (300 held-out episodes per condition): how often and how the robot fails",
                 fontsize=11, color=INK, y=1.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"outcomes.{ext}", dpi=170, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main()
