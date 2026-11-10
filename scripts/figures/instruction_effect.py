"""Exploratory figures (not the pre-registered analysis): where and when the instruction effect changes.

Usage:  python scripts/figures/instruction_effect.py [--data-root DIR] [--pass NAME]
  -> results/instruction_effect_exploratory/<pass>/summary.md, results/figures/instruction_effect_exploratory/<pass>/*.png

Input: the open-loop pass `results/instruction_effect/<pass>` (default: the exploratory pass) and its pre-registered summary.json
(scripts/analysis/instruction_effect.py), for the cell intervals. Per captured state: log IER per condition (vlamech.metrics.open_loop).
Phase of a state, from the CLEAN own-instruction gripper command at that state. The gripper dimension is not
normalized (dataset_statistics action.mask[6] = False): it stays in [0, 1], 1 = open, 0 = closed, and upstream
binarizes it at 0.5 (robot_utils.normalize_gripper_action: 2x - 1, then sign); chunk mean < 0.5 = closing.
Phases: approach (before the first closing chunk), carry (closing), release (open again after a carry). Tasks 0 and 5
never close the gripper (approach only). Every condition and task is shown; nothing is selected.
Figures (one point each; details in captions.md next to them):
  ier_hero.png       one representative state, clean vs the corruption with the largest off-axis effect: the camera
                     view, where the hand would go under three bowl instructions, and every condition on the
                     direction plane (IER_par, IER_perp)
  ier_dose.png       IER per condition with the stratified bootstrap CI (thick) and the tasks-resampled CI (thin)
  ier_direction.png  IER split along the clean task contrast (IER_par) and across it (IER_perp), with IER itself
  ier_tasks.png      IER per task x condition (geometric mean over the task's episodes)
  ier_grasp.png      IER_par aligned to the grasp (first closing chunk), condition x chunk, bowl tasks vs other carries
  ier_chunk.png      inside the 8-step chunk: IER_par per step, and the commanded speed profile
Representative state for ier_hero (a rule, not a choice): task 1 ("put the bowl on the stove"), the first closing
chunk of each exploratory episode; of these, the one whose IER_par under the shown condition is the median.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.config import protocol  # noqa: E402
from vlamech.metrics import open_loop as ol  # noqa: E402
from vlamech.metrics.stats import hierarchical_bootstrap  # noqa: E402
from vlamech.utils.io import atomic_write_bytes  # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
UP, DOWN = "#2a78d6", "#eb6834"  # diverging poles: instruction effect larger (blue) / smaller (orange) than clean
FAMILIES = [("blur", "blur, sigma (px)"), ("noise", "noise, sigma (0-255)"), ("jpeg", "JPEG quality\n(lower = stronger)"),
            ("photometric", "photometric\n(stops)")]
GRIPPER_CLOSED_BELOW = 0.5  # upstream binarization threshold of the raw [0, 1] gripper output
SINGLE = ["matched_bg_box", "bg_patches-0.8", "bg_patches-0.8-exactmi", "target_occlusion"]
BOWL, OTHER_CARRY = [1, 4, 8], [2, 3, 6, 9]  # bowl placement (protocol statistics.bowl_tasks); other object carries
TASK_SHORT = ["open middle drawer", "bowl on stove", "wine on cabinet", "top drawer + bowl", "bowl on cabinet",
              "push plate", "cheese in bowl", "turn on stove", "bowl on plate", "wine on rack"]


def ordered(cells) -> list:
    """Controls and single-level conditions first, then each family from mild to strong (JPEG: high to low quality)."""
    out = [c for c in SINGLE if c in cells]
    for fam, _ in FAMILIES:
        keys = [k for k in cells if k.split("-")[0] == fam]
        out += sorted(keys, key=lambda k: -float(k.split("-")[1]) if fam == "jpeg" else float(k.split("-")[1]))
    return out + [c for c in cells if c not in out]


def load(pass_dir: Path) -> pd.DataFrame:
    """One row per (episode, condition, captured state): log IER, VE, phase."""
    rows = []
    for f in sorted(pass_dir.glob("task*-init*.npz")):
        z = np.load(f, allow_pickle=False)
        conds = [c for c, u in zip(z["conditions"], z["usable"]) if u]
        a = z["actions_norm"][:, z["usable"]]
        c0 = conds.index("identity")
        lr, ve = ol.log_ier(a, c0), ol.ve(a, c0)
        dec = ol.decomposition(a, c0)
        closed = a[:, c0, 0, :, 6].mean(-1) < GRIPPER_CLOSED_BELOW
        first = int(np.argmax(closed)) if closed.any() else len(closed)
        k = np.arange(len(closed))
        phase = np.where(k < first, "approach", np.where(closed, "carry", "release"))
        t, i = (int(x) for x in f.stem.replace("task", "").split("-init"))
        for j, c in enumerate(conds):
            if c != "identity":
                rows += [{"task": t, "init_state": i, "condition": c, "k": int(q), "phase": phase[q],
                          "rel": int(q - first) if closed.any() else None,
                          "log_ier": float(lr[q, j]), "ve": float(ve[q, j]),
                          **{m: float(dec[m][q, j]) for m in dec}} for q in k]
    return pd.DataFrame(rows)


def ci_mean(d: pd.DataFrame, col: str, strata=("task",)) -> tuple:
    r = hierarchical_bootstrap(d, lambda x: x[col].mean(), levels=("init_state",), strata=strata,
                               n_boot=int(protocol("statistics.n_boot")), alpha=float(protocol("statistics.alpha")))
    return r["estimate"], r["ci_low"], r["ci_high"]


def style(ax, plt):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def log_axis(ax, lo, hi):
    ax.set_yscale("log")
    ticks = [t for t in (0.5, 0.67, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(lo, hi)
    ax.axhline(1.0, color=INK, lw=0.8)


def fig_dose(summary, out, plt):
    cells = summary["cells"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), sharey=True, gridspec_kw={"width_ratios": [3, 2, 2, 2, 4.5]})
    lo, hi = 0.8, 2.0

    def draw(ax, xs, labels, keys):
        for x, k in zip(xs, keys):
            e, l, h = np.exp(cells[k]["log_ier"])
            _, sl, sh = np.exp(cells[k]["log_ier_tasks_resampled"])
            ax.plot([x, x], [sl, sh], color=UP, lw=1, alpha=0.5)
            ax.plot([x, x], [l, h], color=UP, lw=4, solid_capstyle="round")
            ax.plot(x, e, "o", color=UP, ms=7, mec=SURF, mew=1.5)
            ax.annotate(f"{e:.2f}", (x, e), xytext=(7, 0), textcoords="offset points", fontsize=7.5, color=INK,
                        va="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.9)

    for ax, (fam, lab) in zip(axes, FAMILIES):
        keys = [k for k in ordered(cells) if k.split("-")[0] == fam]
        draw(ax, list(range(len(keys))), [k.split("-")[1] for k in keys], keys)
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
    draw(axes[4], list(range(len(SINGLE))), ["box\n(control)", "bg\npatches", "bg patches\n(exact input)",
                                           "target\nocclusion"], SINGLE)
    axes[4].set_title("single-level conditions", fontsize=9, color=INK, loc="left")
    for ax in axes:
        style(ax, plt)
        log_axis(ax, lo, hi)
    axes[0].set_ylabel("IER = instruction effect, corrupted / clean\n(log scale; > 1: the instruction moves the "
                       "actions more)", fontsize=8, color=INK)
    fig.suptitle("Exploratory pass (150 episodes, 2,103 states): the instruction effect grows with corruption "
                 "strength", fontsize=10, color=INK, x=0.01, y=1.06, ha="left")
    fig.text(0.01, -0.04, "Point: geometric mean over episodes. Thick bar: 95% CI, tasks fixed, init states resampled "
             "(pre-registered). Thin bar: tasks also resampled.", fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_tasks(df, cells, out, plt):
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    m = np.exp(df.groupby(["task", "init_state", "condition"]).log_ier.mean()
               .groupby(["task", "condition"]).mean().unstack()[cells])
    cmap = LinearSegmentedColormap.from_list("div", [DOWN, "#f3f2ee", UP])
    fig, ax = plt.subplots(figsize=(12, 4.6))
    im = ax.imshow(m.to_numpy(), cmap=cmap, norm=LogNorm(1 / 2.5, 2.5), aspect="auto")
    for (r, c), v in np.ndenumerate(m.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color=SURF if abs(np.log(v)) > 0.6 else INK)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"{t}  {TASK_SHORT[t]}" for t in m.index], fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.yaxis.set_major_locator(plt.FixedLocator([0.5, 0.67, 1, 1.5, 2]))
    cb.ax.yaxis.set_major_formatter(plt.FixedFormatter(["0.5", "0.67", "1", "1.5", "2"]))
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=7)
    cb.set_label("IER (geometric mean over episodes)", fontsize=8)
    ax.set_title("IER per task and condition: most tasks rise, the bowl-placement tasks (1, 4, 8) stay near 1 or fall",
                 fontsize=10, color=INK, loc="left")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_direction(df, cells, out, plt):
    """Per cell: IER (log scale) next to IER_par and IER_perp (linear; metrics.ier_decomposition), episode means."""
    ep = df.groupby(["task", "init_state", "condition"])[["log_ier", "ier_par", "ier_perp"]].mean().reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    y = np.arange(len(cells))
    for i, c in enumerate(cells):
        d = ep[ep.condition == c]
        e, l, h = np.exp(ci_mean(d, "log_ier"))
        a1.plot([l, h], [i, i], color=INK, lw=1.6)
        a1.plot(e, i, "D", color=INK, ms=5, mec=SURF, mew=1)
        for col, mk, m, dy in ((UP, "o", "ier_par", -0.14), (DOWN, "s", "ier_perp", 0.14)):
            e, l, h = ci_mean(d, m)
            a2.plot([l, h], [i + dy] * 2, color=col, lw=1.6)
            a2.plot(e, i + dy, mk, color=col, ms=5.5, mec=SURF, mew=1,
                    label=("along the clean task contrast (IER_par)" if m == "ier_par" else
                           "across it (IER_perp)") if i == 0 else None)
    a1.set_xscale("log")
    a1.set_xticks([0.9, 1, 1.25, 1.5])
    a1.set_xticklabels(["0.9", "1", "1.25", "1.5"])
    a1.minorticks_off()
    a1.axvline(1, color=INK, lw=0.8)
    a1.set_yticks(y)
    a1.set_yticklabels(cells, fontsize=7.5)
    a1.invert_yaxis()
    a1.set_xlabel("IER (log scale): size of the instruction effect", fontsize=8, color=INK)
    a1.set_title("Size: grows with corruption", fontsize=9, color=INK, loc="left")
    a2.axvline(1, color=UP, lw=0.8, ls="--")
    a2.axvline(0, color=DOWN, lw=0.8, ls="--")
    a2.set_xlabel("fraction of the clean instruction effect (linear; 95% CI, tasks fixed)", fontsize=8, color=INK)
    a2.set_title("Direction: less along the clean contrast, more across it", fontsize=9, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    for ax in (a1, a2):
        style(ax, plt)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
    fig.suptitle("The instruction still moves the actions under corruption, but less in the clean task's direction "
                 "(clean: IER_par = 1, IER_perp = 0)", fontsize=10, color=INK, x=0.01, y=1.04, ha="left")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def horizon(pass_dir: Path) -> pd.DataFrame:
    """Per (episode, condition, step h of the 8-step chunk): IER_par(h) and log IER(h) on the 6 arm dims of step h,
    and the commanded speed profile |translation(h)| / chunk mean, own instruction (episode means over states)."""
    rows = []
    for f in sorted(pass_dir.glob("task*-init*.npz")):
        z = np.load(f, allow_pickle=False)
        conds = [c for c, u in zip(z["conditions"], z["usable"]) if u]
        a = z["actions_norm"][:, z["usable"]]
        c0 = conds.index("identity")
        x = a[..., list(protocol("metrics.arm_dims"))]
        d = x[:, :, :1] - x[:, :, 1:-1]  # (q, c, 9, 8, 6)
        n = np.linalg.norm(d, axis=-1)
        n0 = n[:, [c0]]
        par = (d * d[:, [c0]]).sum(-1) / n0
        ie0 = n0.mean(2)
        v = np.linalg.norm(a[:, :, 0, :, :3], axis=-1)
        speed = v / v.mean(-1, keepdims=True)
        t, i = (int(q) for q in f.stem.replace("task", "").split("-init"))
        for j, c in enumerate(conds):
            for h in range(x.shape[3]):
                rows.append({"task": t, "init_state": i, "condition": c, "h": h,
                             "ier_par": float((par[:, j, :, h].mean(1) / ie0[:, 0, h]).mean()),
                             "log_ier": float((np.log(n[:, j, :, h].mean(1)) - np.log(ie0[:, 0, h])).mean()),
                             "speed": float(speed[:, j, h].mean())})
    return pd.DataFrame(rows)


def heat(ax, m, cmap, norm, plt, xlabels, ylabels, title):
    im = ax.imshow(m, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    return im


def diverging(lo, hi):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    return LinearSegmentedColormap.from_list("div", [DOWN, "#f3f2ee", UP]), TwoSlopeNorm(1.0, lo, hi)


def fig_grasp(df, cells, out, plt):
    """IER_par aligned to the first closing chunk; condition x chunk offset; bowl tasks vs other object carries."""
    ep = df[df.rel.notna()].groupby(["task", "init_state", "condition", "rel"]).ier_par.mean().reset_index()
    groups = {"bowl placement (tasks 1, 4, 8)": BOWL, "other object carries (tasks 2, 3, 6, 9)": OTHER_CARRY}
    n_ep = {g: ep[ep.task.isin(t)].groupby("rel")[["task", "init_state"]].apply(
        lambda d: len(d.drop_duplicates())) for g, t in groups.items()}
    rels = sorted(set.intersection(*[set(v[v >= 10].index) for v in n_ep.values()]))  # >= 10 episodes in each group
    cmap, norm = diverging(0.3, 1.7)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (g, tasks) in zip(axes, groups.items()):
        m = (ep[ep.task.isin(tasks) & ep.rel.isin(rels)].groupby(["condition", "rel"]).ier_par.mean()
             .unstack().reindex(index=cells, columns=rels))
        im = heat(ax, m.to_numpy(), cmap, norm, plt, [f"{int(r):+d}" for r in rels], cells, g)
        ax.axvline(rels.index(0) - 0.5, color=INK, lw=1.2)
        ax.set_xlabel("chunks (8 steps) relative to the grasp", fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[0.4, 0.7, 1.0, 1.3, 1.6])
    cb.set_label("IER_par (1 = clean steering)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Right after the grasp, the instruction steers the bowl less", fontsize=10, color=INK,
                 x=0.01, y=1.0, ha="left")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    return rels


def fig_chunk(hz, cells, out, plt):
    """Inside the chunk: IER_par per step h, and the own-instruction speed profile (clean row on top)."""
    ep = hz.groupby(["task", "init_state", "condition", "h"])[["ier_par", "speed"]].mean().reset_index()
    m1 = ep.groupby(["condition", "h"]).ier_par.mean().unstack().reindex(cells)
    m2 = ep.groupby(["condition", "h"]).speed.mean().unstack().reindex(["identity", *cells])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.75})
    c1, n1 = diverging(0.6, 1.4)
    im1 = heat(a1, m1.to_numpy(), c1, n1, plt, [str(h) for h in m1.columns], cells,
               "Steering (IER_par) by step: the far end of the chunk loses most")
    c2, n2 = diverging(0.88, 1.12)
    im2 = heat(a2, m2.to_numpy(), c2, n2, plt, [str(h) for h in m2.columns], ["clean", *cells],
               "Commanded speed / chunk mean: clean speeds up, corrupted slows down")
    a2.axhline(0.5, color=INK, lw=1.2)
    for ax, im, lab in ((a1, im1, "IER_par"), (a2, im2, "speed relative to the chunk's mean")):
        ax.set_xlabel("step inside the 8-step chunk (0 = first executed)", fontsize=8, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(lab, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Inside one action chunk: corruption hits the later steps", fontsize=10, color=INK, x=0.01, y=1.0,
                 ha="left")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_hero(df, cells, pass_dir, capture_dir, out, plt):
    """One representative state (rule in the module docstring), clean vs the condition with the largest IER_perp."""
    from vlamech.envs.offline import recreate

    from vlamech.analysis.sweep import conditions, label_of
    from vlamech.envs.replay import load_capture
    from vlamech.perturbations.image import EpisodeCorruptor

    show = df.groupby("condition").ier_perp.mean().idxmax()
    cand = df[(df.task == 1) & (df.rel == 0) & (df.condition == show)].sort_values("ier_par").reset_index(drop=True)
    pick = cand.iloc[(len(cand) - 1) // 2]
    key = f"task1-init{int(pick.init_state)}"
    q = int(pick.k)
    z = np.load(pass_dir / f"{key}.npz", allow_pickle=False)
    conds = list(z["conditions"])
    acts, ins = z["actions_norm"], list(z["instructions"])
    cap = load_capture(capture_dir / f"{key}.npz")
    spec = next(s for s in conditions() if label_of(s) == show)
    corr = EpisodeCorruptor(spec, key)
    for k in range(q + 1):  # corruptors are called in query order, as in the closed loop
        av, _, _ = recreate(corr, cap, k, center_crop=True)

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.75, 1.5, 1.35], wspace=0.25, hspace=0.18)
    for row, (img, lab, edge) in enumerate(((cap["agentview"][q], "clean", UP), (av, show, DOWN))):
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(np.ascontiguousarray(img[:, ::-1]))  # stored frames are mirrored; this is the true camera view
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(edge)
            sp.set_linewidth(2)
        ax.set_title(f"{'A. ' if row == 0 else ''}{lab}", fontsize=9, color=INK, loc="left")

    ax = fig.add_subplot(gs[:, 1])
    targets = [("put the bowl on the stove", '"…on the stove" (asked)', INK),
               ("put the bowl on top of the cabinet", '"…on top of the cabinet"', "#1baf7a"),
               ("put the bowl on the plate", '"…on the plate"', "#8a5cd0")]  # categorical, not the blue/orange of "more/less"
    ends = []
    for text, lab, colr in targets:
        idx = ins.index(text)
        for c, ls, lw in (("identity", "-", 2.2), (show, "--", 1.6)):
            path = np.vstack([[0, 0], np.cumsum(acts[q, conds.index(c), idx, :, :2], axis=0)])
            ax.plot(path[:, 0], path[:, 1], ls, color=colr, lw=lw, alpha=1 if c == "identity" else 0.85)
            ax.plot(*path[-1], "o" if c == "identity" else "s", color=colr, ms=5, mec=SURF, mew=0.8)
        ends.append([acts[q, conds.index(c), idx, :, :3].sum(0) for c in ("identity", show)])  # 3-D path end
        ax.plot([], [], "-", color=colr, lw=2.2, label=lab)
    ax.plot([], [], "-", color=MUTED, lw=2, label="clean (circle = end)")
    ax.plot([], [], "--", color=MUTED, lw=1.6, label=f"{show} (square = end)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.plot(0, 0, "o", color=INK, ms=4)
    ax.annotate("hand now", (0, 0), xytext=(0, 6), textcoords="offset points", fontsize=7, color=MUTED, ha="center")
    style(ax, plt)
    ax.grid(axis="x", color=GRID, lw=0.6)
    lim = np.array([ln.get_xydata() for ln in ax.lines if len(ln.get_xydata()) > 2], dtype=object)
    pts = np.vstack(list(lim))
    pad = 0.15 * np.ptp(pts, axis=0).max()
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("commanded x over the next 8 steps (normalized)", fontsize=8, color=INK)
    ax.set_ylabel("commanded y (normalized)", fontsize=8, color=INK)
    ax.set_title("B. hand path commanded for 3 instructions", fontsize=9, color=INK, loc="left")

    ax = fig.add_subplot(gs[:, 2])
    ep = df.groupby(["task", "init_state", "condition"])[["ier_par", "ier_perp"]].mean().reset_index()
    ax.plot(1, 0, "o", color=UP, ms=8, mec=SURF, mew=1.2)
    ax.annotate("clean", (1, 0), xytext=(6, -2), textcoords="offset points", fontsize=7.5, color=UP)
    label = {"matched_bg_box", "blur-8", "noise-128", "photometric-4.5", "target_occlusion"}
    for c in cells:
        e = ep[ep.condition == c]
        x0, y0 = e.ier_par.mean(), e.ier_perp.mean()
        ax.plot(x0, y0, "o", color=DOWN if c == show else INK, ms=5.5, mec=SURF, mew=0.8)
        if c in label:
            ax.annotate(c, (x0, y0), xytext=(5, 2), textcoords="offset points", fontsize=7, color=INK)
    ax.axvline(1, color=UP, lw=0.8, ls="--")
    style(ax, plt)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlim(0.55, 1.08)
    ax.set_ylim(-0.08, 1.85)
    ax.set_xlabel("along the clean task contrast (IER_par)", fontsize=8, color=INK)
    ax.set_ylabel("across it (IER_perp)", fontsize=8, color=INK)
    ax.set_title("C. all 13 conditions: less along, more across", fontsize=9, color=INK, loc="left")
    fig.suptitle("Under corruption the instruction still moves the robot, but less toward the asked goal",
                 fontsize=10.5, color=INK, x=0.01, y=1.02, ha="left")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    from itertools import combinations

    gap = [float(np.mean([np.linalg.norm(e1[j] - e2[j]) for e1, e2 in combinations(ends, 2)])) for j in range(2)]
    return {"condition": show, "episode": key, "query": q, "step": int(z["steps"][q]),
            "ier_par": float(pick.ier_par), "n_candidates": len(cand), "gap_clean": gap[0], "gap_corrupt": gap[1]}


def captions(pass_name: str, hero: dict, rels: list) -> str:
    """Figure captions (exploratory pass; numbers that depend on a selection rule are filled in from the data)."""
    return f"""# Captions: instruction effect, `{pass_name}` (EXPLORATORY)

**ier_hero.png.** One state, the instruction effect seen directly.
- *State.* Chosen by rule, not by eye: task 1 ("put the bowl on the stove"), the first closing chunk of each of the
  {hero['n_candidates']} exploratory episodes, the one with the median IER_par under {hero['condition']}. This is
  {hero['episode']}, step {hero['step']}; its IER_par is {hero['ier_par']:.2f}.
- *Condition.* {hero['condition']}, the condition with the largest mean IER_perp.
- (A) The camera view (true orientation) clean and corrupted.
- (B) The hand path the policy commands over the next 8 steps (normalized x-y translation, summed) when told to put
  the bowl on the stove, on the cabinet or on the plate. Solid: clean. Dashed: corrupted. In 3-D (x, y and height)
  the three path ends are on average {hero['gap_clean']:.2f} apart on clean images and {hero['gap_corrupt']:.2f}
  apart under {hero['condition']} (normalized units)."""  + (" Under corruption the three instructions send the hand to nearly the same place."
  if hero["gap_corrupt"] < 0.5 * hero["gap_clean"] else "") + f"""
- (C) Every corrupted condition on the direction plane: IER_par, the instruction effect along the clean task
  contrast (clean = 1), against IER_perp, the effect across it (clean = 0). Exploratory episode means; intervals are
  in ier_direction.png.

**ier_dose.png.** IER per condition, grouped by family and ordered by severity. Points are geometric means over
episodes. Thick bars: 95% stratified bootstrap CI (tasks fixed, init states resampled, as pre-registered). Thin bars:
tasks also resampled.

**ier_direction.png.** Left: the size of the instruction effect (IER, log scale). Right: its direction, as the
fraction of the clean effect along the clean task contrast (IER_par) and across it (IER_perp), linear scale, 95% CIs
(tasks fixed). Even the matched-box control sits slightly below 1 (any change to the image costs ~2% alignment),
so effects are read against that floor.

**ier_tasks.png.** IER per task and condition (geometric mean over the task's 15 episodes); blue = larger than
clean, orange = smaller.

**ier_grasp.png.** IER_par aligned to the grasp.
- Columns: chunks of 8 steps relative to the first chunk in which the clean policy closes the gripper (raw gripper
  output < 0.5); offsets with at least 10 episodes in each group: {', '.join(f'{int(r):+d}' for r in rels)}.
- Left: the bowl-placement tasks (1, 4, 8), where the instruction only chooses the destination.
- Right: the other object carries (2, 3, 6, 9). The stove knob (task 7) is left out: its gripper closes but carries
  nothing.
- Mean over episodes of the per-episode mean at each offset.

**ier_chunk.png.** Inside one 8-step action chunk (step 0 is executed first).
- Left: IER_par computed on each step's 6 arm dimensions.
- Right: the commanded translation speed under the own instruction at each step, divided by the chunk's mean speed;
  the clean row is on top.
- Episode means.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--pass", dest="pass_name", default="identity-agentview+wrist__ins-own__rep0")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pass_dir = a.data_root / "results" / "instruction_effect" / a.pass_name
    summary = json.loads((pass_dir / "summary.json").read_text())
    if not summary.get("exploratory"):
        raise RuntimeError("figures_instruction_effect reads exploratory passes only (the confirmatory one is analysed by "
                           "scripts/analysis/instruction_effect.py as pre-registered)")
    df = load(pass_dir)
    cells = ordered(list(summary["cells"]))
    figs = ROOT / "results" / "figures" / "instruction_effect_exploratory" / a.pass_name
    tabs = ROOT / "results" / "instruction_effect_exploratory" / a.pass_name
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    capture_dir = a.data_root / "results" / "clean_capture" / a.pass_name / "captures"  # exploratory passes only
    hero = fig_hero(df, cells, pass_dir, capture_dir, figs / "ier_hero.png", plt)
    fig_dose(summary, figs / "ier_dose.png", plt)
    fig_direction(df, cells, figs / "ier_direction.png", plt)
    fig_tasks(df, cells, figs / "ier_tasks.png", plt)
    rels = fig_grasp(df, cells, figs / "ier_grasp.png", plt)
    fig_chunk(horizon(pass_dir), cells, figs / "ier_chunk.png", plt)
    cell_ve = df.groupby("condition")[["log_ier", "ve"]].mean()
    r = float(np.corrcoef(cell_ve.ve, cell_ve.log_ier)[0, 1])
    atomic_write_bytes(figs / "captions.md", captions(a.pass_name, hero, rels).encode())

    ep = df.groupby(["task", "init_state", "condition", "phase"]).log_ier.mean()
    by_phase = np.exp(ep.groupby(["task", "init_state", "phase"]).mean().groupby(["task", "phase"]).mean()
                      .unstack())
    n_states = df.groupby("phase").k.size() // len(cells)
    L = ["# Instruction effect, exploratory figures", "",
         f"Pass `{a.pass_name}`, {df[['task', 'init_state']].drop_duplicates().shape[0]} episodes; not the "
         "pre-registered analysis (that is `summary.md` next to the pass). Phases from the clean gripper command; "
         f"captured states per phase: {', '.join(f'{p} {n}' for p, n in n_states.items())}.", "",
         f"Cell IER vs VE: r = {r:.2f} (log IER, 13 cells).", "",
         "## IER by phase per task (all corrupted cells pooled; geometric mean over episodes)", "",
         "| task | " + " | ".join(by_phase.columns) + " |", "|---|" + "---|" * len(by_phase.columns)]
    L += [f"| {t} {TASK_SHORT[t]} | " + " | ".join("" if np.isnan(v) else f"{v:.2f}" for v in row) + " |"
          for t, row in zip(by_phase.index, by_phase.to_numpy())]
    atomic_write_bytes(tabs / "summary.md", ("\n".join(L) + "\n").encode())
    print("\n".join(L))
    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
