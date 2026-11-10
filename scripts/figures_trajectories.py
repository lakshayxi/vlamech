"""Exploratory behaviour figures from the corruption sweep (not the pre-registered analysis).

Usage:  python scripts/figures_trajectories.py [--data-root DIR] [--partial]
  -> results/corruption_sweep_exploratory/{episodes.jsonl, summary.md}, results/figures/corruption_sweep_exploratory/*.png

Inputs: every frozen condition (configs/frozen_levels.yaml), loaded with the strict ingestion (complete runs, clean code,
manifest stamps, trace hashes). --partial skips incomplete conditions and stamps every output as a progress view.
Per episode: oracle v2 outcome (frozen thresholds), the exploratory destination label (protocol
oracle.destination_label), hazard proxies, hand-speed profile inside the 8-step action chunk.
Figures (all conditions, nothing selected by hand):
  trails_task<t>.png   top-down hand paths per condition, coloured by outcome
  barcodes.png         what the robot touches at every step (one row per episode), collisions in black
  chunk_anatomy.png    hand speed and turning angle by position inside the action chunk
  divergence.png       hand distance from the clean twin (same task and init state) over time
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.sweep import build_table  # noqa: E402
from vlamech.utils.io import atomic_write_bytes  # noqa: E402

CHUNK, MAX_STEPS = 8, 300
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
OUTCOME_COL = {"success": "#2a78d6", "intent_loss": "#eb6834", "execution_loss": "#1baf7a",
               "wrong_destination": "#eda100", "no_progress": MUTED}
OUTCOME_LAB = {"success": "success", "intent_loss": "wrong task (v2)", "execution_loss": "right task, botched",
               "wrong_destination": "wrong destination", "no_progress": "stalled"}
TOUCH = ["nothing", "own target", "other object", "fixture", "table"]
TOUCH_COL = ["#f3f2ee", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def summary(rows, skipped, partial) -> str:
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    out = ["# Corruption sweep, exploratory behaviour figures", "",
           ("**PROGRESS VIEW: incomplete conditions skipped: " + ", ".join(skipped) + "**\n") if partial and skipped else "",
           "Outcome v2 = frozen oracle (confirmatory). Destination-aware = v2, except failures whose manipulated object "
           "headed for another task's destination (protocol oracle.destination_label).", "",
           "| condition | n | success | wrong task (v2) | right task, botched (v2) | of which wrong destination | stalled | "
           "collision | off table |", "|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        v2 = Counter(r["outcome_v2"] for r in rs)
        wd = sum(r["outcome_dest"] == "wrong_destination" for r in rs)
        wd_ex = sum(r["outcome_dest"] == "wrong_destination" and r["outcome_v2"] == "execution_loss" for r in rs)
        out.append(f"| {c} | {len(rs)} | {v2['success']} | {v2['intent_loss']} | {v2['execution_loss']} | {wd_ex} "
                   f"(all: {wd}) | {v2['no_progress']} | {sum(r['collision'] for r in rs)} | "
                   f"{sum(r['off_table'] for r in rs)} |")
    return "\n".join(out) + "\n"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)


def fig_trails(traces, conds, task, out, plt):
    eps = {c: [(k, v) for (cc, k), v in traces.items() if cc == c and k[0] == task] for c in conds}
    allp = np.concatenate([v["eef"] for c in conds for _, v in eps[c]])
    lo, hi = allp.min(0), allp.max(0)
    ncol = 7
    nrow = -(-len(conds) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.6 * nrow + 0.6), facecolor=SURF, squeeze=False)
    for ax, c in zip(axes.flat, conds):
        for _, v in sorted(eps[c], key=lambda e: e[1]["outcome"] != "success"):
            e = v["eef"]
            ok = v["outcome"] == "success"
            ax.plot(e[:, 1], e[:, 0], color=OUTCOME_COL[v["outcome"]], lw=0.6 if ok else 0.9, alpha=0.45 if ok else 0.9)
        n_ok = sum(v["outcome"] == "success" for _, v in eps[c])
        ax.set_title(f"{c}  ({n_ok}/{len(eps[c])})", fontsize=8, color=INK)
        ax.set_xlim(hi[1] + 0.02, lo[1] - 0.02), ax.set_ylim(lo[0] - 0.02, hi[0] + 0.02)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        [s.set_color(GRID) for s in ax.spines.values()]
    for ax in list(axes.flat)[len(conds):]:
        ax.axis("off")
    h = [plt.Line2D([], [], color=OUTCOME_COL[k], lw=2, label=OUTCOME_LAB[k]) for k in OUTCOME_COL]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Task {task}: top-down hand paths, every held-out episode, every condition (exploratory)",
                 y=1.04, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_barcodes(traces, conds, out, plt):
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(TOUCH_COL + [INK])
    ncol = 7
    nrow = -(-len(conds) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 3.6 * nrow + 0.6), facecolor=SURF, squeeze=False)
    order = list(OUTCOME_COL)
    for ax, c in zip(axes.flat, conds):
        eps = sorted([(k, v) for (cc, k), v in traces.items() if cc == c],
                     key=lambda e: (e[0][0], order.index(e[1]["outcome"]), e[0][1]))
        img = np.full((len(eps), MAX_STEPS), np.nan)
        for i, (_, v) in enumerate(eps):
            code = v["touch"].astype(float)
            code[v["collide"]] = len(TOUCH)
            img[i, :len(code)] = code
        ax.imshow(img, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=len(TOUCH) + 0.5)
        for i in range(1, 10):
            ax.axhline(i * len(eps) / 10 - 0.5, color=SURF, lw=0.6)
        ax.set_title(c, fontsize=8, color=INK)
        ax.set_yticks([(i + 0.5) * len(eps) / 10 for i in range(10)], [f"t{i}" for i in range(10)])
        ax.set_xticks([0, 100, 200, 300])
        _style(ax)
    for ax in list(axes.flat)[len(conds):]:
        ax.axis("off")
    h = [plt.Rectangle((0, 0), 1, 1, color=col, label=lab) for lab, col in zip(TOUCH + ["collision"], TOUCH_COL + [INK])]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("What the robot touches at every control step: one row per episode (grouped by task, sorted by outcome); "
                 "white = episode over (exploratory)", y=1.04, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def chunk_profile(vs):
    """Speed share and turning angle by position inside the chunk, pooled over all complete chunks after the first."""
    tot, ang, n_ang = np.zeros(CHUNK), np.zeros(CHUNK), np.zeros(CHUNK)
    for v, e in vs:
        m = (len(v) // CHUNK) * CHUNK
        if m <= CHUNK:
            continue
        tot += v[CHUNK:m].reshape(-1, CHUNK).sum(0)
        d = np.diff(e, axis=0)[:m]
        cos = np.einsum("ij,ij->i", d[1:], d[:-1]) / (np.linalg.norm(d[1:], axis=1) * np.linalg.norm(d[:-1], axis=1) + 1e-12)
        a = np.degrees(np.arccos(np.clip(cos, -1, 1)))  # a[i] = turn between step i and i+1
        idx = np.arange(1, m)[CHUNK - 1:]
        w = v[1:m][CHUNK - 1:]
        np.add.at(ang, idx % CHUNK, a[CHUNK - 1:] * w)
        np.add.at(n_ang, idx % CHUNK, w)
    return tot / tot.mean(), ang / np.maximum(n_ang, 1e-12)


def fig_chunks(traces, conds, out, plt):
    fams = list(dict.fromkeys(c.split("-")[0] for c in conds if c != "identity"))
    fig, axes = plt.subplots(2, len(fams), figsize=(2.6 * len(fams), 5.2), facecolor=SURF, squeeze=False)
    base = chunk_profile([(v["speed"], v["eef"]) for (c, _), v in traces.items() if c == "identity"])
    blues = ["#9ec5f0", "#5a9be3", "#2a78d6", "#1b4f8f"]
    for j, fam in enumerate(fams):
        levels = [c for c in conds if c.split("-")[0] == fam]
        for i, (row, ylab) in enumerate([(0, "speed / chunk mean"), (1, "turning angle (deg, speed-weighted)")]):
            ax = axes[i, j]
            ax.plot(range(CHUNK), base[row], color=MUTED, lw=1.2, ls="--", label="clean")
            for li, c in enumerate(levels):
                prof = chunk_profile([(v["speed"], v["eef"]) for (cc, _), v in traces.items() if cc == c])
                ax.plot(range(CHUNK), prof[row], color=blues[min(li + (4 - len(levels)), 3)], lw=1.8, marker="o", ms=3,
                        label=c)
            ax.set_xticks(range(CHUNK))
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8, color=INK)
            if i == 1:
                ax.set_xlabel("step inside the 8-step action chunk", fontsize=8, color=INK)
            if i == 0:
                ax.set_title(fam, fontsize=9, color=INK)
                ax.legend(fontsize=6, frameon=False)
            _style(ax)
    fig.suptitle("Inside the action chunk: speed decays and the hand turns at chunk boundaries under corruption "
                 "(all episodes, all chunks after the first; exploratory)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_divergence(traces, conds, out, plt):
    cs = [c for c in conds if c != "identity"]
    ncol = 7
    nrow = -(-len(cs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow + 0.5), facecolor=SURF, squeeze=False,
                             sharey=True)
    for ax, c in zip(axes.flat, cs):
        for oc in OUTCOME_COL:
            ds = []
            for (cc, k), v in traces.items():
                if cc != c or v["outcome"] != oc:
                    continue
                a, b = v["eef"], traces[("identity", k)]["eef"]
                n = MAX_STEPS + 1
                pa = np.vstack([a, np.repeat(a[-1:], n - len(a), 0)])  # an episode that ended stays where it stopped
                pb = np.vstack([b, np.repeat(b[-1:], n - len(b), 0)])
                ds.append(np.linalg.norm(pa - pb, axis=1) * 100)
            if len(ds) >= 3:
                ax.plot(np.median(ds, 0), color=OUTCOME_COL[oc], lw=1.6, label=f"{OUTCOME_LAB[oc]} ({len(ds)})")
        ax.set_title(c, fontsize=8, color=INK)
        ax.legend(fontsize=5.5, frameon=False)
        _style(ax)
    for ax in list(axes.flat)[len(cs):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("hand distance from clean twin (cm)", fontsize=8, color=INK)
    fig.suptitle("When does an episode leave its clean twin (same task, same init state)? Median per outcome "
                 "(exploratory; the clean run is not bit-reproducible, so small distances are not divergence)",
                 y=1.03, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT, help="checkout whose results/ holds the sweep (default: this one)")
    ap.add_argument("--partial", action="store_true", help="skip incomplete conditions (progress view only)")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, traces, skipped, dest = build_table(a.data_root, a.partial)
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    out = ROOT / "results" / "corruption_sweep_exploratory"
    fig_dir = ROOT / "results" / "figures" / "corruption_sweep_exploratory"
    out.mkdir(parents=True, exist_ok=True), fig_dir.mkdir(parents=True, exist_ok=True)
    stamp = {"partial": a.partial, "skipped": skipped,
             "destinations": {f"{o}|{t}": [round(float(x), 4) for x in mu] for (o, t), mu in dest.items()}}
    atomic_write_bytes(out / "episodes.jsonl",
                       ("\n".join(json.dumps(r) for r in [{"_meta": stamp}] + rows) + "\n").encode())
    md = summary(rows, skipped, a.partial)
    atomic_write_bytes(out / "summary.md", md.encode())
    for t in range(10):
        fig_trails(traces, conds, t, fig_dir / f"trails_task{t}.png", plt)
    fig_barcodes(traces, conds, fig_dir / "barcodes.png", plt)
    fig_chunks(traces, conds, fig_dir / "chunk_anatomy.png", plt)
    fig_divergence(traces, conds, fig_dir / "divergence.png", plt)
    print(md)


if __name__ == "__main__":
    main()
