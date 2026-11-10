"""Oracle validation by eye: sampled corruption-sweep episodes as keyframe strips with the oracle's labels, plus a verdict sheet.

Usage:  python scripts/figures/oracle_review.py --per-class N [--seed S] [--partial] [--data-root DIR]
  -> results/oracle_review/<seed>/{sheet_<class>_<page>.png, review.csv}

Sampling (stratified, seeded, decided before looking): for each oracle class (v2: success, intent_loss,
execution_loss, no_progress; exploratory: wrong_destination) N episodes drawn uniformly without replacement from all
sweep episodes of that class (all of them if fewer). Each episode is shown as 6 frames of its video at equal time steps
(the video shows the scene, both robot inputs, the step and the goals holding), with the oracle labels. review.csv
has one row per sampled episode with empty `verdict` / `notes` columns for the reviewer: verdict = agree | disagree
(+ what actually happened).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.sweep import build_table, conditions, label_of  # noqa: E402
from vlamech.utils.seeding import derive_seed  # noqa: E402

CLASSES = ["success", "intent_loss", "execution_loss", "no_progress", "wrong_destination"]
PER_PAGE, FRAMES = 8, 6
SCENE = (slice(56, 312), slice(8, 264))  # the "scene (true camera view)" panel of a RolloutVideo frame (vlamech.viz.video)


def frames(video: Path, n: int = FRAMES) -> list:
    import imageio.v2 as imageio

    r = imageio.get_reader(video)
    total = r.count_frames()
    idx = np.linspace(0, total - 1, n).round().astype(int)
    out = [r.get_data(int(i))[SCENE] for i in idx]
    r.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--data-root", type=Path, default=ROOT)
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, _, skipped, _ = build_table(a.data_root, a.partial)
    run_dir = {label_of(s): a.data_root / "results" / "corruption_sweep" / f"{s.tag}__ins-own__rep0" for s in conditions()}
    out = ROOT / "results" / "oracle_review" / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    picked = []
    for cls in CLASSES:
        key = "outcome_dest" if cls == "wrong_destination" else "outcome_v2"
        cand = [r for r in rows if r[key] == cls]
        rng = np.random.default_rng(derive_seed("oracle-review", cls, a.seed))
        take = rng.choice(len(cand), size=min(a.per_class, len(cand)), replace=False) if cand else []
        picked += [(cls, cand[i]) for i in sorted(take)]
    with open(out / "review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "condition", "task", "init", "outcome_v2", "subtype_v2", "outcome_dest", "heading",
                    "collision", "verdict", "notes"])
        for cls, r in picked:
            w.writerow([cls, r["condition"], r["task"], r["init"], r["outcome_v2"], r["subtype_v2"], r["outcome_dest"],
                        r["heading"], r["collision"], "", ""])
    for cls in CLASSES:
        eps = [r for c, r in picked if c == cls]
        for p in range(0, len(eps), PER_PAGE):
            page = eps[p:p + PER_PAGE]
            fig, ax = plt.subplots(len(page), FRAMES, figsize=(FRAMES * 2.3, len(page) * 2.6 + 0.4), squeeze=False)
            fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.5 / (len(page) * 2.6 + 0.4), bottom=0.01,
                                wspace=0.03, hspace=0.18)
            for i, r in enumerate(page):
                imgs = frames(run_dir[r["condition"]] / "videos" / f"task{r['task']}-init{r['init']}.mp4")
                for j, im in enumerate(imgs):
                    ax[i, j].imshow(im)
                    ax[i, j].axis("off")
                lab = (f"{r['condition']}  task{r['task']}-init{r['init']}  |  v2: {r['outcome_v2']}"
                       f"{' (' + r['subtype_v2'] + ')' if r['subtype_v2'] else ''}  |  destination label: "
                       f"{r['outcome_dest']}{'' if r['heading'] is None else ' -> task ' + str(r['heading'])}"
                       f"{'  |  COLLISION' if r['collision'] else ''}")
                ax[i, 0].text(0.0, 1.03, lab, transform=ax[i, 0].transAxes, fontsize=8, va="bottom", ha="left")
            fig.suptitle(f"Oracle review: {cls} (page {p // PER_PAGE + 1}; scene view, 6 frames per episode at equal "
                         "time steps; full videos in the run's videos/)",
                         fontsize=10)
            fig.savefig(out / f"sheet_{cls}_{p // PER_PAGE + 1}.png", dpi=110)
            plt.close(fig)
    print(f"{len(picked)} episodes sampled -> {out}" + (f" (PROGRESS VIEW: skipped {skipped})" if skipped else ""))


if __name__ == "__main__":
    main()
