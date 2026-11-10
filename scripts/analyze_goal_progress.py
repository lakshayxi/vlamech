"""Physical check of the instruction metric on the bowl tasks: do commanded hand motions head for the instructed destination?

Usage:  python scripts/analyze_goal_progress.py [--data-root DIR]  -> results/goal_progress/{summary.json, summary.md},
        results/figures/goal/goal_margin.png

Post hoc and not pre-registered (added 2026-11-09). Held-out open-loop actions (clean repeat rep1) on tasks in
statistics.bowl_tasks, at captured states where the bowl is held (clean own-instruction gripper chunk mean < 0.5).
For each state, condition and each of the three bowl instructions: the commanded 8-step displacement in the horizontal
plane (unnormalized with the checkpoint statistics, 0.05 m per unit, the OSC controller's output_max), starting at the
hand. The destinations are the mean final bowl positions of clean successes of each task.
  progress(dest)  = distance to dest before the chunk - distance after it
  margin          = progress(instructed destination) - mean progress(the two other destinations)
  picked          = the instructed destination has the largest progress
Translation-only IER_par uses the same states and instructions. Validity check: on clean images the margin should be
positive and the instructed destination usually picked.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.sweep import build_table  # noqa: E402
from vlamech.config import model_revision, protocol  # noqa: E402
from vlamech.metrics.stats import hierarchical_bootstrap  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json, read_json  # noqa: E402

OSC_METRES_PER_UNIT = 0.05  # robosuite osc_pose.json output_max for translation
GRIPPER_CLOSED_BELOW = 0.5
BLUE, MUTED, GRID, SURF = "#2a78d6", "#898781", "#e1e0d9", "#fcfcfb"
ORDER = ["identity", "matched_bg_box", "blur-1", "blur-4", "blur-8", "noise-64", "noise-128", "jpeg-6", "jpeg-1",
         "photometric-3.5", "photometric-4.5", "bg_patches-0.8", "bg_patches-0.8-exactmi", "target_occlusion"]


def checkpoint_file(name: str) -> Path:
    """A file of the pinned OFT checkpoint from the local Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    repo = protocol("policy.checkpoint")
    return Path(hf_hub_download(repo, name, revision=model_revision(repo), local_files_only=True))


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
    from libero.libero import benchmark

    tasks = sorted(protocol("statistics.bowl_tasks"))
    suite = benchmark.get_benchmark_dict()[protocol("env.suite")]()
    lang = {t: suite.get_task(t).language for t in tasks}
    dest_all = build_table(a.data_root, False)[3]
    bowl = [k for k in dest_all if k[1] == tasks[0]][0][0]
    dest = {t: np.asarray(dest_all[(bowl, t)])[:2] for t in tasks}
    pass_dir = a.data_root / "results" / "instruction_effect" / "identity-agentview+wrist__ins-own__rep1"
    capt = a.data_root / "results" / "corruption_sweep" / "identity-agentview+wrist__ins-own__rep1" / "captures"
    s = read_json(checkpoint_file("dataset_statistics.json"))["libero_goal_no_noops"]["action"]
    lo, hi = np.asarray(s["q01"][:2], np.float64), np.asarray(s["q99"][:2], np.float64)

    rows = []
    n_states = 0
    for t in tasks:
        for f in sorted(pass_dir.glob(f"task{t}-init*.npz")):
            z = np.load(f, allow_pickle=False)
            init = int(f.stem.split("init")[1])
            conds, ins, usable = list(z["conditions"]), list(z["instructions"]), z["usable"]
            acts = z["actions_norm"].astype(np.float64)
            hand = np.load(capt / f"{f.stem}.npz", allow_pickle=False)["proprio"][:, :2]
            if len(hand) != len(acts):
                raise RuntimeError(f"{f.stem}: {len(hand)} captured states, {len(acts)} open-loop states")
            c0 = conds.index("identity")
            idx = [ins.index(lang[u]) for u in tasks]
            own = ins.index(lang[t])
            held = acts[:, c0, own, :, 6].mean(-1) < GRIPPER_CLOSED_BELOW
            if not held.any():
                continue
            n_states += int(held.sum())
            xy = 0.5 * (acts[held][:, :, idx, :, :2] + 1) * (hi - lo) + lo
            disp = xy.sum(-2) * OSC_METRES_PER_UNIT  # (states, conditions, 3, 2), metres
            e = hand[held][:, None, None, :]
            # translation-only contrast along the clean one, own instruction against the other two
            tr = acts[held][:, :, idx, :, :3].reshape(*acts[held][:, :, idx, :, :3].shape[:-2], -1)
            o = idx.index(own)
            alts = [i for i in range(3) if i != o]
            d_c = tr[:, :, [o]] - tr[:, :, alts]
            d0 = d_c[:, [c0]]
            n0 = np.linalg.norm(d0, axis=-1)
            par = np.where(n0 > 0, (d_c * d0).sum(-1) / np.where(n0 > 0, n0, 1.0), 0.0).mean(-1) / n0.mean(-1)
            mus = np.stack([dest[u] for u in tasks])  # 3 x 2
            before = np.linalg.norm(e[..., None, :] - mus[None, None, None], axis=-1)  # (states, 1, 1, 3)
            after = np.linalg.norm((e + disp)[..., None, :] - mus[None, None, None], axis=-1)  # (states, conds, 3, 3)
            progress = before - after  # (states, conds, instruction, destination)
            margin = progress[:, :, range(3), range(3)] - (progress.sum(-1) - progress[:, :, range(3), range(3)]) / 2
            picked = progress.argmax(-1) == np.arange(3)[None, None]
            for j, c in enumerate(conds):
                if usable[j]:
                    rows.append({"task": t, "init_state": init, "condition": c, "margin": float(margin[:, j].mean()) * 100,
                                 "picked": float(picked[:, j].mean()), "par_translation": float(par[:, j].mean())})
    d = pd.DataFrame(rows)
    wide = d.pivot_table(index=["task", "init_state"], columns="condition", values="margin").reset_index()
    out = {"n_episodes": int(d[["task", "init_state"]].drop_duplicates().shape[0]), "n_states": n_states, "cells": {}}
    order = [c for c in ORDER if c in set(d.condition)]
    for c in order:
        x = d[d.condition == c]
        cell = {"margin_cm": boot(x, lambda v: v["margin"].mean()), "picked": boot(x, lambda v: v["picked"].mean()),
                "par_translation": boot(x, lambda v: v["par_translation"].mean())}
        if c != "identity":
            cell["margin_ratio"] = boot(wide, lambda v: v[c].mean() / v["identity"].mean())
        out["cells"][c] = cell
    corrupted = [c for c in order if c != "identity"]
    out["spearman_ratio_vs_par_translation"] = float(sps.spearmanr([out["cells"][c]["margin_ratio"][0] for c in corrupted],
                                                                   [out["cells"][c]["par_translation"][0] for c in corrupted]).statistic)
    f = lambda v, p=2: f"{v[0]:.{p}f} [{v[1]:.{p}f}, {v[2]:.{p}f}]"  # noqa: E731
    L = ["# Physical check on the bowl tasks (post hoc, not pre-registered; held-out, open loop)", "",
         f"{out['n_episodes']} episodes, {n_states} states with the bowl held. Margin: commanded 8-step progress toward the instructed destination "
         "minus progress toward the other two, horizontal plane, cm per chunk. Picked: the instructed destination gets the most progress.", "",
         "| condition | margin (cm) | margin vs clean | picked | translation along clean (bowl contrast) |", "|---|---|---|---|---|"]
    for c in order:
        x = out["cells"][c]
        L.append(f"| {c} | {f(x['margin_cm'])} | {f(x['margin_ratio']) if 'margin_ratio' in x else '1'} | {f(x['picked'])} | {f(x['par_translation'])} |")
    L += ["", f"Spearman across corrupted conditions, margin ratio against translation along clean: {out['spearman_ratio_vs_par_translation']:.2f}."]
    base = a.data_root / "results" / "goal_progress"
    base.mkdir(parents=True, exist_ok=True)
    atomic_write_json(base / "summary.json", out)
    atomic_write_bytes(base / "summary.md", ("\n".join(L) + "\n").encode())
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for i, c in enumerate(corrupted):
        e, l, h = out["cells"][c]["margin_ratio"]
        ax.plot([l, h], [i, i], color=BLUE, lw=2, zorder=1)
        ax.scatter(e, i, s=50, color=BLUE, zorder=2)
    ax.axvline(1, color=MUTED, lw=0.8, ls="--")
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_yticks(range(len(corrupted)))
    ax.set_yticklabels(corrupted, fontsize=8)
    ax.set_xlabel("progress toward the instructed destination relative to clean (1 = clean; 95% CI)", fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    figs = a.data_root / "results" / "figures" / "goal"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "goal_margin.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
