"""Held-out instruction effect: size (IER) against direction (IER_par, IER_perp) per condition, drawn from the
pre-registered summary (scripts/analysis/instruction_effect.py). No statistic is computed here.

Usage:  python scripts/figures/instruction_effect_heldout.py
  -> results/figures/instruction_effect/size_and_direction.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PASS = ROOT / "results" / "instruction_effect" / "identity-agentview+wrist__ins-own__rep1"
OUT = ROOT / "results" / "figures" / "instruction_effect" / "size_and_direction.png"
ORDER = ["matched_bg_box", "bg_patches-0.8", "bg_patches-0.8-exactmi", "target_occlusion", "blur-1", "blur-4", "blur-8",
         "noise-64", "noise-128", "jpeg-6", "jpeg-1", "photometric-3.5", "photometric-4.5"]
INK, GRID, SURF, ALONG, ACROSS = "#0b0b0b", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = json.loads((PASS / "summary.json").read_text())["cells"]
    order = [c for c in ORDER if c in cells]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for i, c in enumerate(order):
        e, lo, hi = (math.exp(v) for v in cells[c]["log_ier"])
        a1.plot([lo, hi], [i, i], color=INK, lw=1.6)
        a1.plot(e, i, "D", color=INK, ms=5, mec=SURF, mew=1)
        for key, col, mk, dy, lab in (("ier_par", ALONG, "o", -0.14, "along the clean task contrast (IER_par)"),
                                      ("ier_perp", ACROSS, "s", 0.14, "across it (IER_perp)")):
            e, lo, hi = cells[c][key]
            a2.plot([lo, hi], [i + dy] * 2, color=col, lw=1.6)
            a2.plot(e, i + dy, mk, color=col, ms=5.5, mec=SURF, mew=1, label=lab if i == 0 else None)
    a1.set_xscale("log")
    a1.set_xticks([0.9, 1, 1.25, 1.5])
    a1.set_xticklabels(["0.9", "1", "1.25", "1.5"])
    a1.minorticks_off()
    a1.axvline(1, color=INK, lw=0.8)
    a1.set_yticks(range(len(order)))
    a1.set_yticklabels(order, fontsize=8)
    a1.invert_yaxis()
    a1.set_xlabel("IER: size of the instruction effect relative to clean (log scale)", fontsize=8)
    a1.set_title("Size: larger under corruption", fontsize=9, loc="left")
    a2.axvline(1, color=ALONG, lw=0.8, ls="--")
    a2.axvline(0, color=ACROSS, lw=0.8, ls="--")
    a2.set_xlabel("fraction of the clean instruction effect (95% CI)", fontsize=8)
    a2.set_title("Direction: less along the clean contrast, more across it", fontsize=9, loc="left")
    a2.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_facecolor(SURF)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(labelsize=8)
    fig.suptitle("Held-out (300 episodes, about 4,350 states): clean is IER = 1, IER_par = 1, IER_perp = 0",
                 fontsize=10, x=0.01, y=1.03, ha="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
