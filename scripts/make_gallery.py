"""Figures of what the robot sees: the input pipeline, every corruption family, and the segmentation masks.

Usage:  python scripts/make_gallery.py [--task "put the bowl on the plate"] [--init 0]
Output: results/figures/gallery/{pipeline,corruptions_agentview,corruptions_wrist,masks}.{png,pdf}

Every image is produced by the same code the experiments use: LiberoTaskEnv.reset (upstream frame + fixture
layout), EpisodeCorruptor (the study's corruptions, ladders from configs/protocol.yaml), and
vlamech.models.oft.policy_view (the exact upstream resize + crop the network receives). No model weights are
needed. Severity columns show ladder levels evenly spread from the mildest to the strongest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "figures" / "gallery"
UNITS = {
    "photometric": lambda v: f"-{v:g} stops",
    "noise": lambda v: f"σ = {v:g}",
    "blur": lambda v: f"σ = {v:g} px",
    "jpeg": lambda v: f"quality {v:g}",
    "bg_patches": lambda v: f"{v:.0%} of background",
}
TITLES = {
    "photometric": "Exposure (darker)", "noise": "Sensor noise", "blur": "Defocus blur", "jpeg": "JPEG compression",
    "bg_patches": "Background patches", "target_occlusion": "Target occluded\n(task-relevant control)",
    "matched_bg_box": "Same-size box on background\n(area-matched control)",
}
N_COLS = 4


def task_id_for(language: str) -> int:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    ids = [i for i in range(suite.n_tasks) if suite.get_task(i).language == language]
    if len(ids) != 1:
        raise ValueError(f"no unique LIBERO-Goal task {language!r}")
    return ids[0]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", OUT / f"{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="put the bowl on the plate")
    ap.add_argument("--init", type=int, default=0)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vlamech.envs.libero_env import LiberoTaskEnv
    from vlamech.models.oft import policy_view
    from vlamech.perturbations.assets import bg_side_lengths
    from vlamech.perturbations.image import LADDER_KEY, CorruptionSpec, EpisodeCorruptor, ladder
    from experiments.robot.openvla_utils import resize_image_for_policy

    task_id = task_id_for(args.task)
    envs = {}

    def make_env(t):
        if t not in envs:
            envs[t] = LiberoTaskEnv("libero_goal", t, segmentation=True)
        return envs[t]

    env = make_env(task_id)
    sides = bg_side_lengths(make_env)
    frame = env.reset(args.init)
    masks = env.masks(frame)
    key = f"task{task_id}-init{args.init}"
    caption = f'"{env.instruction}"  (LIBERO-Goal task {task_id}, init state {args.init})'

    # ---------------------------------------------------------------- 1. input pipeline
    fig, ax = plt.subplots(2, 3, figsize=(9.5, 6.6))
    for r, cam in enumerate(("agentview", "wrist")):
        raw = getattr(frame, cam)
        resized = resize_image_for_policy(np.array(raw), 224)
        seen = np.asarray(policy_view(raw))
        for c, (img, t) in enumerate([(raw, "camera frame 256×256\n(rotated 180° as upstream)"),
                                      (resized, "JPEG round-trip +\nLanczos resize to 224"),
                                      (seen, "center crop (90% area) → 224\n= what the network receives")]):
            ax[r, c].imshow(img)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            ax[r, c].set_title(t, fontsize=9)
        ax[r, 0].set_ylabel({"agentview": "third-person camera", "wrist": "wrist camera"}[cam], fontsize=10)
    fig.suptitle("What the robot sees: from simulator camera to network input\n" + caption, fontsize=10)
    save(fig, "pipeline")
    plt.close(fig)

    # ---------------------------------------------------------------- 2. corruption grid (per camera)
    rows = []
    for fam in LADDER_KEY:
        lad = ladder(fam)
        idx = np.unique(np.round(np.linspace(0, len(lad) - 1, N_COLS)).astype(int))
        rows.append((fam, [CorruptionSpec(fam, lad[i]) for i in idx]))
    rows.append(("controls", [CorruptionSpec("target_occlusion"), CorruptionSpec("matched_bg_box")]))
    for cam in ("agentview", "wrist"):
        fig, ax = plt.subplots(len(rows), N_COLS + 1, figsize=(2.1 * (N_COLS + 1), 2.25 * len(rows)))
        for r, (fam, specs) in enumerate(rows):
            ax[r, 0].imshow(np.asarray(policy_view(getattr(frame, cam))))
            ax[r, 0].set_title("clean", fontsize=8)
            ax[r, 0].set_ylabel(TITLES.get(fam, "Controls"), fontsize=9)
            for c in range(N_COLS):
                a = ax[r, c + 1]
                if c < len(specs):
                    spec = specs[c]
                    corr = EpisodeCorruptor(spec, key, sides)
                    img = corr(cam, getattr(frame, cam), step=0, background_mask=masks[cam]["background"],
                               target_mask=masks[cam]["target"])
                    a.imshow(np.asarray(policy_view(img)))
                    a.set_title(UNITS[spec.family](spec.value) if spec.family in UNITS else TITLES[spec.family], fontsize=8)
                else:
                    a.axis("off")
            for a in ax[r]:
                a.set_xticks([]), a.set_yticks([])
        cam_name = {"agentview": "third-person camera", "wrist": "wrist camera"}[cam]
        fig.suptitle(f"Implemented corruptions, {cam_name}, as the network receives them (224×224)\n{caption}\n"
                     "columns: mildest → strongest level of each family's ladder (operating points are chosen later "
                     "by calibration)", fontsize=10)
        save(fig, f"corruptions_{cam}")
        plt.close(fig)

    # ---------------------------------------------------------------- 3. masks
    fig, ax = plt.subplots(2, 3, figsize=(9.5, 6.6))
    for r, cam in enumerate(("agentview", "wrist")):
        img = getattr(frame, cam).astype(float) / 255
        bg, tgt = masks[cam]["background"], masks[cam]["target"]
        over = img.copy()
        over[bg] = 0.45 * over[bg] + 0.55 * np.array([0.15, 0.35, 0.95])
        over[tgt] = 0.45 * over[tgt] + 0.55 * np.array([0.95, 0.2, 0.15])
        patched = EpisodeCorruptor(CorruptionSpec("bg_patches", ladder("bg_patches")[-2]), key, sides)(
            cam, getattr(frame, cam), 0, background_mask=bg, target_mask=tgt)
        for c, (im, t) in enumerate([(getattr(frame, cam), "camera frame"),
                                     (over, "blue = background (may be corrupted)\nred = this task's objects\n"
                                            "untinted = robot + every other task's objects"),
                                     (patched, f"background patches ({ladder('bg_patches')[-2]:.0%}):\n"
                                               "composited behind robot and objects")]):
            ax[r, c].imshow(im)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            ax[r, c].set_title(t, fontsize=8)
        ax[r, 0].set_ylabel({"agentview": "third-person camera", "wrist": "wrist camera"}[cam], fontsize=10)
    fig.suptitle("Task-irrelevance by construction: background is defined by simulator segmentation\n" + caption, fontsize=10)
    save(fig, "masks")
    plt.close(fig)
    for e in envs.values():
        e.close()


if __name__ == "__main__":
    main()
