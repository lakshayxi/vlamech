"""Failure monitor and the safe stop (protocol mitigation.monitor_*, mitigation.safe_stop_rule; H4).

Usage:  python scripts/analyze_failure_monitor.py [--data-root DIR]  -> results/failure_monitor/summary.{json,md}, results/figures/failure_monitor/*.png

Development, exploratory split, plain arm (cag:1) of the mitigation-selection runs:
  init states 5-12: per-task normalization (mean and SD of each signal over clean successful queries) and the choice of
                    signal (highest failure AUROC of the early-window score);
  init states 13-19: the safe-stop threshold (split conformal on the whole-episode maximum of clean successful episodes).
Test, confirmatory init states, conditions no earlier analysis has seen (mitigation.monitor_test_conditions).
Early-window score: the maximum over the task's median clean episode length in queries, so longer failing episodes are
not favoured by having more queries. Safe stop: applied offline (stopping only ends the episode before the action, so
the executed prefix is exactly what would have happened). Unsafe failure: failure with a collision above the task's
clean maximum before the episode ends or is stopped. Timing section (post hoc, added 2026-11-09): when the alert comes
relative to the first collision, precision and recall of the stop for failure, average precision of each signal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.ingest import trace  # noqa: E402
from vlamech.analysis.sweep import touch_codes  # noqa: E402
from vlamech.analysis.arms import commanded, load_outcomes, thresholds  # noqa: E402
from vlamech.config import protocol, split_states  # noqa: E402
from vlamech.metrics import monitor as M  # noqa: E402
from vlamech.metrics.stats import auroc, hierarchical_bootstrap, holm, mcnemar_exact  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

PLAIN = "cag:1"
INK, MUTED, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"


def collect(experiment: str, labels: list, states: list, thr: dict) -> list:
    eps = []
    for c in labels:
        rd, df = load_outcomes(ROOT, experiment, c, PLAIN, states, thr)
        for row in df.to_dict("records"):
            key = (row["task"], row["init_state"])
            acts, steps, ie = commanded(rd, key)
            eps.append({**row, "condition": c, "query_steps": steps, "signals": M.query_signals(acts, ie), "rd": rd, "key": key})
    return eps


def normalize(eps: list, stats: dict) -> None:
    for e in eps:
        e["z"] = {s: (e["signals"][s] - stats[(e["task"], s)][0]) / stats[(e["task"], s)][1] for s in M.SIGNALS}


def add_scores(eps: list, lengths: dict) -> None:
    for e in eps:
        L = lengths[e["task"]]
        for s in M.SIGNALS:
            z = e["z"][s]
            e[f"early|{s}"] = float(np.nanmax(z[:L]))
            e[f"full|{s}"] = float(np.nanmax(z))


def boot(df: pd.DataFrame, stat, alpha: float) -> list:
    r = hierarchical_bootstrap(df, stat, levels=tuple(protocol("statistics.bootstrap_levels")),
                               strata=tuple(protocol("statistics.bootstrap_strata")),
                               n_boot=int(protocol("statistics.n_boot")), alpha=alpha)
    return [r["estimate"], r["ci_low"], r["ci_high"]]


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ROOT = ap.parse_args().data_root
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr, alpha = thresholds(ROOT), float(protocol("statistics.alpha"))
    fit_labels = list(protocol("mitigation.monitor_fit_conditions"))
    test_labels = list(protocol("mitigation.monitor_test_conditions"))
    dev_states, cal_states = [list(range(a, b + 1)) for a, b in protocol("mitigation.monitor_split")]
    if sorted(dev_states + cal_states) != split_states("exploratory"):
        raise RuntimeError("mitigation.monitor_split must partition the exploratory split")
    fit = collect("mitigation_selection", fit_labels, split_states("exploratory"), thr)
    dev = [e for e in fit if e["init_state"] in dev_states]
    cal = [e for e in fit if e["init_state"] in cal_states]
    clean_dev = [e for e in dev if e["condition"] == "identity" and e["success"]]
    stats, lengths = {}, {}
    for t in range(10):
        ep = [e for e in clean_dev if e["task"] == t]
        lengths[t] = int(np.median([len(e["query_steps"]) for e in ep]))
        for s in M.SIGNALS:
            v = np.concatenate([e["signals"][s] for e in ep])
            stats[(t, s)] = (float(np.nanmean(v)), float(np.nanstd(v)))
    test = collect("failure_monitor", test_labels, split_states("confirmatory"), thr)
    for group in (dev, cal, test):
        normalize(group, stats)
        add_scores(group, lengths)

    dev_df = pd.DataFrame(dev)
    dev_auc = {s: auroc(dev_df[f"early|{s}"].to_numpy(), (~dev_df.success).to_numpy()) for s in M.SIGNALS}
    chosen = max(M.SIGNALS, key=lambda s: dev_auc[s])
    clean_cal = np.array([e[f"full|{chosen}"] for e in cal if e["condition"] == "identity" and e["success"]])
    alphas = [float(x) for x in protocol("mitigation.safe_stop_alphas")]
    thresholds_z = {a: M.conformal_threshold(clean_cal, a) for a in alphas}

    td = pd.DataFrame([{k: v for k, v in e.items() if k not in ("query_steps", "signals", "rd", "key", "z")} for e in test])
    out = {"dev_auroc": dev_auc, "chosen": chosen, "n_calibration": int(len(clean_cal)), "thresholds": {str(k): v for k, v in thresholds_z.items()},
           "lengths": lengths, "test_failure_rate": float((~td.success).mean())}
    # ---- AUROC on the held-out set, pooled over test conditions; instruction effect vs each baseline
    out["auroc"] = {}
    for s in M.SIGNALS:
        out["auroc"][s] = boot(td, lambda x: auroc(x[f"early|{s}"].to_numpy(), (~x.success).to_numpy()), alpha)
    others = [s for s in M.SIGNALS if s != "instruction_effect"]
    out["ie_minus"] = {}
    for s in others:
        out["ie_minus"][s] = boot(td, lambda x: auroc(x["early|instruction_effect"].to_numpy(), (~x.success).to_numpy())
                                  - auroc(x[f"early|{s}"].to_numpy(), (~x.success).to_numpy()), alpha / len(others))

    # ---- safe stop, offline
    L_stop, frames = {}, {}
    for a, thr_z in thresholds_z.items():
        rows = []
        for e in test:
            q = M.first_exceedance(e["z"][chosen], thr_z)
            stop = None if q is None else int(e["query_steps"][q])
            tr = trace(e["rd"], e["key"])
            collide = touch_codes(tr, e["task"], thr["force"][str(e["task"])]["environment"])[1]
            first = int(np.argmax(collide)) if collide.any() else None
            without = (not e["success"]) and first is not None
            stopped = stop is not None and stop < e["steps"]
            withstop = without and (not stopped or first < stop)
            rows.append({"condition": e["condition"], "task": e["task"], "init_state": e["init_state"], "success": e["success"],
                         "stopped": stopped, "success_lost": e["success"] and stopped, "unsafe_without": without, "unsafe_with": withstop,
                         "stop_step": stop if stopped else np.nan, "first_collision": first if first is not None else np.nan})
        d = pd.DataFrame(rows)
        frames[a] = d
        table, ps, keys = {}, [], []
        for c in test_labels:
            x = d[d.condition == c]
            t = mcnemar_exact(x.unsafe_without.to_numpy(), x.unsafe_with.to_numpy())
            table[c] = {"n": int(len(x)), "failure": float((~x.success).mean()), "stopped": float(x.stopped.mean()),
                        "success_lost": float(x.success_lost.mean()), "unsafe_without": float(x.unsafe_without.mean()),
                        "unsafe_with": float(x.unsafe_with.mean()), "p": t["p"]}
            if c != "identity":
                ps.append(t["p"])
                keys.append(c)
        for c, p in zip(keys, holm(ps)):
            table[c]["p_holm"] = float(p)
        L_stop[a] = table
    out["safe_stop"] = {str(a): t for a, t in L_stop.items()}
    out["off_table_episodes"] = int(td.off_table.sum())
    a0 = alphas[0]
    t0 = frames[a0]
    corrupted_only = t0[t0.condition != "identity"]
    unsafe = corrupted_only[corrupted_only.unsafe_without]
    before = unsafe[unsafe.stop_step.notna() & (unsafe.stop_step <= unsafe.first_collision)]
    leads = (before.first_collision - before.stop_step).to_numpy()
    fail = ~corrupted_only.success
    timing = {"unsafe_events": int(len(unsafe)), "caught_before_collision": int(len(before)),
              "lead_steps_median": float(np.median(leads)) if len(leads) else None,
              "lead_steps_range": [float(leads.min()), float(leads.max())] if len(leads) else None,
              "precision_for_failure": float((fail & corrupted_only.stopped).sum() / corrupted_only.stopped.sum()),
              "recall_of_failure": float((fail & corrupted_only.stopped).sum() / fail.sum()),
              "base_failure_rate": float(fail.mean()),
              "stop_step_median_failures": float(corrupted_only[fail & corrupted_only.stopped].stop_step.median()),
              "average_precision": {s: M.average_precision(td[f"early|{s}"].to_numpy(), (~td.success).to_numpy()) for s in M.SIGNALS},
              "base_rate_test": float((~td.success).mean())}
    out["timing"] = timing

    f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"  # noqa: E731
    L = ["# Failure monitor and safe stop", "",
         f"Signal chosen on init states 5-12: **{chosen}** (development AUROC {dev_auc[chosen]:.3f}). Calibrated on {len(clean_cal)} "
         f"clean successful episodes (init states 13-19). Test: {len(td)} episodes, failure rate {out['test_failure_rate']:.3f}.", "",
         "## Failure AUROC on the held-out conditions (early window)", "", "| signal | dev AUROC | test AUROC [CI] |", "|---|---|---|"]
    L += [f"| {s} | {dev_auc[s]:.3f} | {f(out['auroc'][s])} |" for s in M.SIGNALS]
    L += ["", f"Instruction effect minus each other signal (CI at alpha {alpha}/{len(others)}):", ""]
    L += [f"- {s}: {f(out['ie_minus'][s])}" for s in others]
    for a in alphas:
        L += ["", f"## Safe stop on {chosen}, alpha = {a:g} (threshold z = {thresholds_z[a]:.2f})", "",
              "| condition | failure | stopped | success lost | unsafe without | unsafe with | p_holm |", "|---|---|---|---|---|---|---|"]
        for c in test_labels:
            x = L_stop[a][c]
            p = f"{x['p_holm']:.3g}" if "p_holm" in x else f"{x['p']:.3g}"
            L.append(f"| {c} | {x['failure']:.3f} | {x['stopped']:.3f} | {x['success_lost']:.3f} | {x['unsafe_without']:.3f} | {x['unsafe_with']:.3f} | {p} |")
    tm = out["timing"]
    L += ["", f"## Timing of the alert (alpha = {alphas[0]:g}, corrupted conditions; post hoc)", "",
          f"- Unsafe failures without the stop: {tm['unsafe_events']}. Alert at or before the first collision: {tm['caught_before_collision']}; "
          f"lead in steps: median {tm['lead_steps_median']}, range {tm['lead_steps_range']}.",
          f"- Stop as a failure detector: precision {tm['precision_for_failure']:.3f}, recall {tm['recall_of_failure']:.3f} (failure rate {tm['base_failure_rate']:.3f}); "
          f"median stop step on failures {tm['stop_step_median_failures']:.0f}.",
          "", "| signal | average precision | test failure rate |", "|---|---|---|"]
    L += [f"| {s} | {tm['average_precision'][s]:.3f} | {tm['base_rate_test']:.3f} |" for s in M.SIGNALS]
    base = ROOT / "results" / "failure_monitor"
    base.mkdir(parents=True, exist_ok=True)
    atomic_write_json(base / "summary.json", out)
    atomic_write_bytes(base / "summary.md", ("\n".join(L) + "\n").encode())

    figs = ROOT / "results" / "figures" / "failure_monitor"
    figs.mkdir(parents=True, exist_ok=True)
    order = sorted(M.SIGNALS, key=lambda s: out["auroc"][s][0])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, s in enumerate(order):
        e, lo, hi = out["auroc"][s]
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2, zorder=1)
        ax.scatter(e, i, s=60, color=BLUE, zorder=2)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("failure AUROC on held-out conditions (95% CI)", fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(figs / "auroc.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    a0 = alphas[0]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(test_labels))
    ax.barh(y + 0.18, [L_stop[a0][c]["unsafe_without"] for c in test_labels], 0.34, color=MUTED, label="no stop")
    ax.barh(y - 0.18, [L_stop[a0][c]["unsafe_with"] for c in test_labels], 0.34, color=ORANGE, label=f"safe stop (alpha {a0:g})")
    ax.set_yticks(y)
    ax.set_yticklabels(test_labels, fontsize=8)
    ax.set_xlabel("unsafe failures (share of episodes)", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(figs / "safe_stop.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
