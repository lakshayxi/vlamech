"""Activation patching: verdicts and figures from scripts/run_activation_patching.py outputs (protocol patching.*).

Usage:  python scripts/analyze_activation_patching.py --split exploratory|confirmatory [--data-root DIR]
  -> results/activation_patching/<split>/{summary.json, summary.md}, results/figures/activation_patching/<split>/*.png (+ captions.md)

Metric (patching.metric): bowl IER_par of a set of actions under the 3 bowl instructions relative to the clean actions under
them (vlamech.metrics.open_loop.ier_par_against); restoration rho = (IER_par_patched - IER_par_corrupted) /
(1 - IER_par_corrupted), on states whose corrupted IER_par < 1. Episode = mean over its states; cell = mean over
episodes; stratified cluster bootstrap CI (tasks fixed). Verdict (patching.verdicts, fixed 2026-11-08 before any held-out output), per
condition of patching.patch_conditions, Holm over conditions x the three tests. "Instruction route" needs all three:
  (a) rho of the instruction prefix clamp to layer 31 exceeds its random-donor control;
  (b) corrupted instruction states injected into the clean run (prefix to layer 31) give IER_par below 1;
  (c) they damage it more than clean instruction states of another episode do (noising_control).
Otherwise "not supported". Keeping vision positions clean at every depth is the clean run by construction, so the visual
route is not tested by a clamp. Camera and entity shares are descriptive; the null control has no loss to share out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.config import protocol  # noqa: E402
from vlamech.metrics import open_loop as ol  # noqa: E402
from vlamech.metrics.stats import hierarchical_bootstrap, holm, paired_sign_flip_test  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
UP, DOWN = "#2a78d6", "#eb6834"
BOWL_TASKS = sorted(protocol("statistics.bowl_tasks"))
TESTS = ("instr_minus_donor", "noising_minus_1", "noising_minus_donor")
DIRECTIONS = (("corr", "corrupted", DOWN, "o"), ("denoise", "instruction positions kept clean", UP, "s"),
              ("noise", "corrupted instruction states in a clean run", "#8a5cd0", "^"))


def ids(name: str) -> tuple:
    t, rest = name[:-4].split("-init")
    i, q = rest.split("-q")
    return int(t[4:]), int(i), int(q)


def ci(df: pd.DataFrame, col: str) -> list:
    d = df.groupby(["task", "init_state"])[col].mean().reset_index()
    r = hierarchical_bootstrap(d, lambda x: x[col].mean(), levels=tuple(protocol("statistics.bootstrap_levels")),
                               strata=tuple(protocol("statistics.bootstrap_strata")),
                               n_boot=int(protocol("statistics.n_boot")), alpha=float(protocol("statistics.alpha")))
    return [r["estimate"], r["ci_low"], r["ci_high"]]


def episode_p(df: pd.DataFrame, col: str) -> float:
    d = df.groupby(["task", "init_state"])[col].mean().to_numpy()
    return paired_sign_flip_test(d, n_perm=int(protocol("statistics.n_perm")))


def load_patch(d: Path, ctrl_d: Path) -> tuple:
    """Rows per (state, condition of patching.patch_conditions): corrupted IER_par, rho for every (mode, group, point),
    controls, noising, the noising control, and the direction split of the three runs of DIRECTIONS."""
    rows, meta, keep = [], None, set(protocol("patching.patch_conditions"))
    for f in sorted(d.glob("task*-q*.npz")):
        z = np.load(f, allow_pickle=False)
        nc = np.load(ctrl_d / f.name, allow_pickle=False)
        t, i, q = ids(f.name)
        conds, grid = list(z["conditions"]), list(z["grid"])
        meta = {k: list(z[k]) for k in ("groups", "prefix_groups", "grid", "control_points")}
        meta["conditions"] = [c for c in conds if c in keep]
        ins_g = meta["prefix_groups"].index("instruction")
        for c_i, c in enumerate(conds):
            if c not in keep:
                continue
            clean = z["clean"][c_i]
            p_corr = float(ol.ier_par_against(z["corr"][c_i], clean))
            r = {"task": t, "init_state": i, "q": q, "condition": c, "par_corr": p_corr}

            def rho(acts):
                return (float(ol.ier_par_against(acts, clean)) - p_corr) / (1 - p_corr) if p_corr < 1 else np.nan
            for g_i, g in enumerate(meta["groups"]):
                for p_i, p in enumerate(grid):
                    r[f"single|{g}|{p}"] = rho(z["single"][c_i, g_i, p_i])
            for g_i, g in enumerate(meta["prefix_groups"]):
                for p_i, p in enumerate(grid):
                    r[f"prefix|{g}|{p}"] = rho(z["prefix"][c_i, g_i, p_i])
                for p_i, p in enumerate(meta["control_points"]):
                    r[f"donor|{g}|{p}"] = rho(z["ctrl"][c_i, 0, g_i, p_i])
                    r[f"noise|{g}|{p}"] = rho(z["ctrl"][c_i, 1, g_i, p_i])
            for p_i, p in enumerate(grid):
                r[f"noising_single|{p}"] = float(ol.ier_par_against(z["noising"][c_i, 0, p_i], clean))
                r[f"noising_prefix|{p}"] = float(ol.ier_par_against(z["noising"][c_i, 1, p_i], clean))
            for p_i, p in enumerate(nc["control_points"]):
                r[f"noising_donor|{p}"] = float(ol.ier_par_against(nc["noising_control"][p_i], nc["clean"]))
            runs = {"corr": z["corr"][c_i], "denoise": z["prefix"][c_i, ins_g, -1], "noise": z["noising"][c_i, 1, -1]}
            for name, acts in runs.items():
                for k, v in ol.contrast_against(acts, clean).items():
                    r[f"{name}|{k}"] = float(v)
            rows.append(r)
    return pd.DataFrame(rows), meta


def bowl_index(instructions: list, task: int, suite_lang: list) -> list:
    """Indices of [own, other bowl tasks] in an open-loop instruction list (own, 9 others, empty)."""
    return [0] + [instructions.index(suite_lang[t]) for t in BOWL_TASKS if t != task]


def load_inputs(d_cam: Path, d_ent: Path, suite_lang: list) -> pd.DataFrame:
    """Rows per (bowl state, condition, variant): bowl IER_par relative to clean and the 9-alternative IER_par."""
    rows = []
    for f in sorted(d_cam.glob("task*-q*.npz")):
        z = np.load(f, allow_pickle=False)
        t, i, q = ids(f.name)
        labels, ins, acts = list(z["variants"]), list(z["instructions"]), z["actions"]
        clean = acts[labels.index("identity")]
        bi = bowl_index(ins, t, suite_lang) if bool(z["bowl"]) else None
        ent = None
        if bi is not None and (d_ent / f.name).exists():
            e = np.load(d_ent / f.name, allow_pickle=False)
            ent = dict(zip(list(e["parts"]), e["actions"]))
        for lab, a in zip(labels, acts):
            if lab == "identity":
                continue
            c, v = lab.split("|")
            dec = ol.decomposition(np.stack([clean, a])[None], 0)
            rows.append({"task": t, "init_state": i, "q": q, "condition": c, "variant": v, "bowl": bi is not None,
                         "ier_par9": float(dec["ier_par"][0, 1]),
                         "par_bowl": float(ol.ier_par_against(a[bi], clean[bi])) if bi else np.nan})
        if ent is not None:
            for part, a in ent.items():
                rows.append({"task": t, "init_state": i, "q": q, "condition": "target_occlusion", "variant": part,
                             "bowl": True, "ier_par9": float(ol.decomposition(np.stack([clean, a])[None], 0)["ier_par"][0, 1]),
                             "par_bowl": float(ol.ier_par_against(a[bi], clean[bi]))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["exploratory", "confirmatory"], required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT)
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from libero.libero import benchmark
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    suite = benchmark.get_benchmark_dict()[protocol("env.suite")]()
    lang = [suite.get_task(t).language for t in range(suite.n_tasks)]
    base = a.data_root / "results" / "activation_patching" / a.split
    figs = a.data_root / "results" / "figures" / "activation_patching" / a.split
    figs.mkdir(parents=True, exist_ok=True)
    alpha = float(protocol("statistics.alpha"))
    out, L = {"split": a.split}, [f"# Activation patching ({a.split})", ""]
    caps = [f"# Captions: activation patching, {a.split}", ""]

    # ---------------------------------------------------------------- patching
    if any((base / "patch").glob("*.npz")):
        df, meta = load_patch(base / "patch", base / "noising_control")
        grid, conds = meta["grid"], meta["conditions"]
        last = grid[-1]
        out["patch"] = {"n_states": int(df[["task", "init_state", "q"]].drop_duplicates().shape[0]), "cells": {}}
        ps, keys = [], []
        for c in conds:
            d = df[(df.condition == c) & (df.par_corr < 1)].copy()
            d["instr_minus_donor"] = d[f"prefix|instruction|{last}"] - d[f"donor|instruction|{last}"]
            d["noising_minus_1"] = d[f"noising_prefix|{last}"] - 1.0
            d["noising_minus_donor"] = d[f"noising_prefix|{last}"] - d[f"noising_donor|{last}"]
            cell = {"n_states": int(len(d)), "par_corr": ci(df[df.condition == c], "par_corr")}
            for col in (*TESTS, f"prefix|instruction|{last}", f"donor|instruction|{last}", f"noising_prefix|{last}",
                        f"noising_donor|{last}"):
                cell[col] = ci(d, col)
            for col in TESTS:
                cell[f"p_{col}"] = episode_p(d, col)
                ps.append(cell[f"p_{col}"])
                keys.append((c, col))
            for name, *_ in DIRECTIONS:
                for k in ("ier_par", "ier_perp", "cos_align"):
                    cell[f"{name}|{k}"] = ci(d, f"{name}|{k}")
            out["patch"]["cells"][c] = cell
        for (c, col), p in zip(keys, holm(ps)):
            out["patch"]["cells"][c][f"p_holm_{col}"] = float(p)
        for c in conds:
            x = out["patch"]["cells"][c]
            above = x["instr_minus_donor"][1] > 0 and x["p_holm_instr_minus_donor"] < alpha
            below = all(x[t][2] < 0 and x[f"p_holm_{t}"] < alpha for t in ("noising_minus_1", "noising_minus_donor"))
            x["verdict"] = "instruction route" if above and below else "not supported"
        f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
        L += ["## Patching (bowl states)", "",
              "rho = fraction of the corrupted run's loss of bowl steering (IER_par) restored by keeping the instruction "
              f"positions clean up to layer {last}. Injected = the clean run's IER_par with instruction states clamped in "
              "up to that layer: the corrupted run's, or the clean states of another episode (control).", "",
              "| condition | corrupted IER_par | rho instruction | rho random donor | instr - donor | injected: corrupted "
              "| injected: other episode | corrupted - other | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in conds:
            x = out["patch"]["cells"][c]
            L.append(f"| {c} | {f(x['par_corr'])} | {f(x[f'prefix|instruction|{last}'])} | {f(x[f'donor|instruction|{last}'])} "
                     f"| {f(x['instr_minus_donor'])} | {f(x[f'noising_prefix|{last}'])} | {f(x[f'noising_donor|{last}'])} "
                     f"| {f(x['noising_minus_donor'])} | {x['verdict']} |")
        L += ["", "### Direction of the steering in the three runs", "",
              "Along the clean contrast (IER_par, 1 on clean), across it (IER_perp, 0 on clean), cosine to it.", "",
              "| condition | run | IER_par | IER_perp | cos_align |", "|---|---|---|---|---|"]
        for c in conds:
            x = out["patch"]["cells"][c]
            for name, label, *_ in DIRECTIONS:
                L.append(f"| {c} | {label} | {f(x[f'{name}|ier_par'])} | {f(x[f'{name}|ier_perp'])} | {f(x[f'{name}|cos_align'])} |")

        # figure: heatmaps of single-layer rho, one panel per condition
        cmap = LinearSegmentedColormap.from_list("div", [DOWN, "#f3f2ee", UP])
        fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.4), sharey=True)
        for ax, c in zip(np.atleast_1d(axes), conds):
            d = df[(df.condition == c) & (df.par_corr < 1)]
            ep = d.groupby(["task", "init_state"]).mean(numeric_only=True)
            m = np.array([[ep[f"single|{g}|{p}"].mean() for p in grid] for g in meta["groups"]])
            im = ax.imshow(m, cmap=cmap, norm=TwoSlopeNorm(0, -0.5, 1.0), aspect="auto")
            ax.set_xticks(range(len(grid)))
            ax.set_xticklabels([p if p in ("embed", "8", "16", "24", last) else "" for p in grid], fontsize=7)
            ax.set_yticks(range(len(meta["groups"])))
            ax.set_yticklabels(meta["groups"], fontsize=8)
            ax.set_title(c, fontsize=9, loc="left")
            ax.tick_params(length=0)
            ax.set_xlabel("patched at (input, then decoder layer)", fontsize=8)
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01).set_label("rho (1 = clean steering restored)", fontsize=8)
        fig.suptitle("Which positions, at which depth, carry the lost steering? (single-layer patch, bowl states)",
                     fontsize=10, x=0.01, ha="left", y=1.03)
        fig.savefig(figs / "patch_heatmap.png", dpi=140, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)

        # figure: prefix clamps with controls, and noising
        fig, axes = plt.subplots(2, len(conds), figsize=(4.2 * len(conds), 6.2), sharex=True)
        x_of = {p: i for i, p in enumerate(grid)}
        for j, c in enumerate(conds):
            d = df[(df.condition == c) & (df.par_corr < 1)]
            ep = d.groupby(["task", "init_state"]).mean(numeric_only=True)
            ax = axes[0, j]
            cp = meta["control_points"]
            for g, col in (("instruction", DOWN),):
                ax.plot(range(len(grid)), [ep[f"prefix|{g}|{p}"].mean() for p in grid], color=col, lw=2, label=f"{g} clamp")
                ax.plot([x_of[p] for p in cp], [ep[f"donor|{g}|{p}"].mean() for p in cp], "o", color=col, mfc="none",
                        ms=5, label=f"{g}: random donor")
                ax.plot([x_of[p] for p in cp], [ep[f"noise|{g}|{p}"].mean() for p in cp], "x", color=col, ms=5,
                        label=f"{g}: matched noise")
            ax.axhline(0, color=INK, lw=0.6)
            ax.axhline(1, color=MUTED, lw=0.6, ls="--")
            ax.set_title(c, fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel("rho (prefix clamp)", fontsize=8)
                ax.legend(frameon=False, fontsize=6.5)
            ax2 = axes[1, j]
            dd = df[df.condition == c].groupby(["task", "init_state"]).mean(numeric_only=True)
            ax2.plot(range(len(grid)), [dd[f"noising_prefix|{p}"].mean() for p in grid], color=DOWN, lw=2,
                     label="prefix")
            ax2.plot(range(len(grid)), [dd[f"noising_single|{p}"].mean() for p in grid], color=DOWN, lw=1, ls="--",
                     label="single layer")
            ax2.plot([x_of[p] for p in cp], [dd[f"noising_donor|{p}"].mean() for p in cp], "o", color=UP, mfc="none", ms=5,
                     label="clean states of another episode")
            ax2.axhline(1, color=MUTED, lw=0.6, ls="--")
            ax2.set_xticks(range(len(grid)))
            ax2.set_xticklabels([p if p in ("embed", "8", "16", "24", last) else "" for p in grid], fontsize=7)
            ax2.set_xlabel("depth", fontsize=8)
            if j == 0:
                ax2.set_ylabel("clean-run IER_par with corrupted\ninstruction states (1 = unaffected)", fontsize=8)
                ax2.legend(frameon=False, fontsize=7)
        fig.suptitle("Top: keeping the instruction positions clean up to a depth. Bottom: injecting instruction "
                     "states into a clean run", fontsize=10, x=0.01, ha="left", y=1.01)
        fig.savefig(figs / "clamps.png", dpi=140, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
        fig, axes = plt.subplots(1, len(conds), figsize=(3.8 * len(conds), 3.6), sharex=True, sharey=True)
        for j, (ax, c) in enumerate(zip(np.atleast_1d(axes), conds)):
            x = out["patch"]["cells"][c]
            ax.plot(1, 0, "D", color=INK, ms=7, label="clean")
            for name, label, col, mk in DIRECTIONS:
                (px, lx, hx), (py, ly, hy) = x[f"{name}|ier_par"], x[f"{name}|ier_perp"]
                ax.errorbar(px, py, xerr=[[px - lx], [hx - px]], yerr=[[py - ly], [hy - py]], fmt=mk, color=col, ms=8,
                            mec=SURF, lw=1, label=label)
            ax.set_xlim(0, 1.1)
            ax.set_ylim(-0.03, None)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.set_title(c, fontsize=9, loc="left")
            ax.set_xlabel("steering along the clean contrast (IER_par)", fontsize=8)
            if j == 0:
                ax.set_ylabel("steering across it (IER_perp)", fontsize=8)
        fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=4,
                   loc="lower center", bbox_to_anchor=(0.5, -0.08))
        fig.suptitle("Where the instruction positions act: muting versus rotation of bowl steering", fontsize=10, x=0.01,
                     ha="left", y=1.02)
        fig.savefig(figs / "direction.png", dpi=140, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
        caps += ["**direction.png.** The three-way contrast between bowl instructions, split along the clean contrast "
                 "and across it (95% CIs). Clean sits at (1, 0). Moving left is muting; moving up is rotation.", "",
                 "**patch_heatmap.png.** Single-layer denoising on bowl states: the clean run's state of one token "
                 "group at one depth replaces the corrupted run's; colour = rho, the fraction of the lost bowl steering "
                 "(IER_par under the three bowl instructions) restored, episode means. Groups are positions, not "
                 "modalities: after layer 0 each position mixes in all others.", "",
                 "**clamps.png.** Top: prefix clamps (the group kept clean at every depth up to the point) for the "
                 "instruction positions, with the random-donor (same task, other episode, same phase) and "
                 "norm-matched-noise controls. Bottom: noising, the clean run with the corrupted run's instruction "
                 "states clamped in, and the same with another episode's clean states (circles); 1 = steering "
                 "unaffected.", ""]

    # ---------------------------------------------------------------- inputs (camera, entity)
    if any((base / "camera").glob("*.npz")):
        di = load_inputs(base / "camera", base / "entity", lang)
        b = di[di.bowl]
        out["inputs"] = {}
        L += ["", "## Single camera and single entity (bowl states)", "",
              "fraction = (1 - IER_par variant) / (1 - IER_par full condition), bowl IER_par; episode means with CIs.", "",
              "| condition | full IER_par | wrist only | agentview only | bowl only | destination only |", "|---|---|---|---|---|---|"]
        for c in protocol("patching.conditions"):
            full = b[(b.condition == c) & (b.variant == "full")].set_index(["task", "init_state", "q"]).par_bowl
            row = {"full": ci(b[(b.condition == c) & (b.variant == "full")], "par_bowl")}
            if c not in protocol("patching.patch_conditions"):  # the null control has no loss of steering to share out
                out["inputs"][c] = row
                L.append(f"| {c} | {row['full'][0]:.2f} [{row['full'][1]:.2f}, {row['full'][2]:.2f}] | no loss to share out | | | |")
                continue
            for v in ("wrist", "agentview", "bowl", "destination"):
                x = b[(b.condition == c) & (b.variant == v)].set_index(["task", "init_state", "q"])
                if not len(x):
                    continue
                x = x.assign(frac=(1 - x.par_bowl) / (1 - full.reindex(x.index))).reset_index()
                x = x[np.isfinite(x.frac)]
                row[v] = ci(x, "frac")
            out["inputs"][c] = row
            f = lambda k: f"{row[k][0]:.2f} [{row[k][1]:.2f}, {row[k][2]:.2f}]" if k in row else "-"  # noqa: E731
            L.append(f"| {c} | {f('full')} | {f('wrist')} | {f('agentview')} | {f('bowl')} | {f('destination')} |")
        boxed = [np.load(f, allow_pickle=False) for f in sorted((base / "entity").glob("*.npz"))]
        if boxed and "box_pixels" in boxed[0].files:
            mean_box = np.stack([z["box_pixels"] for z in boxed]).mean(0)  # [camera][target, bowl, destination]
            out["entity_box_pixels"] = {"agentview": mean_box[0].tolist(), "wrist": mean_box[1].tolist()}
            L += ["", "Mean box area hidden, pixels (whole target / bowl only / destination only): main camera "
                  f"{mean_box[0, 0]:.0f} / {mean_box[0, 1]:.0f} / {mean_box[0, 2]:.0f}; wrist "
                  f"{mean_box[1, 0]:.0f} / {mean_box[1, 1]:.0f} / {mean_box[1, 2]:.0f}."]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        vs = [("wrist", "wrist camera only", UP), ("agentview", "main camera only", "#1baf7a"),
              ("bowl", "bowl hidden only", DOWN), ("destination", "destination hidden only", "#8a5cd0")]
        conds = protocol("patching.conditions")
        for k, (v, lab, col) in enumerate(vs):
            xs = [i + (k - 1.5) * 0.18 for i in range(len(conds))]
            for x0, c in zip(xs, conds):
                if v in out["inputs"][c]:
                    e, lo, hi = out["inputs"][c][v]
                    ax.bar(x0, e, 0.16, color=col, label=lab if c == conds[0] or v in ("bowl", "destination") and c == "target_occlusion" else None)
                    ax.plot([x0, x0], [lo, hi], color=INK, lw=1)
        ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, fontsize=8)
        ax.set_ylabel("share of the full condition's\nsteering loss reproduced", fontsize=8)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=7.5, ncol=2)
        ax.set_title("Which input carries the damage? (bowl states, 95% CI)", fontsize=10, loc="left")
        fig.savefig(figs / "inputs.png", dpi=140, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
        caps += ["**inputs.png.** On bowl states, the share of each condition's loss of bowl steering that is "
                 "reproduced when only one camera is corrupted, or (target occlusion) when only the bowl or only the "
                 "destination is hidden; 1 = all of it. The bowl/destination split partitions the captured target "
                 "mask by the segmentation of the restored simulator state; each part then hides its own bounding "
                 "box, so the two parts neither cover nor equal the box of the whole target.", ""]

    atomic_write_json(base / "summary.json", json.loads(json.dumps(out, default=float)))
    atomic_write_bytes(base / "summary.md", ("\n".join(L) + "\n").encode())
    atomic_write_bytes(figs / "captions.md", ("\n".join(caps) + "\n").encode())
    print("\n".join(L))


if __name__ == "__main__":
    main()
