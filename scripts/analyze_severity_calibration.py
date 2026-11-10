"""Severity calibration: success vs severity per family, and the severity selection by the protocol rule.

Usage:  python scripts/analyze_severity_calibration.py  ->  results/severity_calibration/{summary.json, summary.md}, results/figures/severity_calibration/success_vs_severity.png

Input: the clean calibration run (identity) and one run per ladder level (scripts/run_severity_calibration.sh), each loaded with
the strict ingestion (complete, clean code, records stamped with their run's manifest). Output: per level success
with Wilson 95% CI, the isotonic fit, the levels selected for each target (vlamech.metrics.selection), and the
families that need the one-step ladder extension or are benign. This script proposes; it does not write
configs/frozen_levels.yaml; scripts/freeze_levels.py does, once the oracle thresholds are validated.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.ingest import find_run, grid, load_run  # noqa: E402
from vlamech.config import protocol, split_states  # noqa: E402
from vlamech.metrics.selection import next_level, select  # noqa: E402
from vlamech.metrics.stats import wilson_ci  # noqa: E402
from vlamech.perturbations.image import LADDER_KEY, CorruptionSpec, ladder  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

STAGE = ROOT / "results" / "severity_calibration"
FIG = ROOT / "results" / "figures" / "severity_calibration"
UNITS = {"photometric": "stops darker", "noise": "noise σ (0-255)", "blur": "blur σ (px)", "jpeg": "JPEG quality",
         "bg_patches": "background coverage"}


def load(tag: str, keys: set):
    rd = load_run(find_run(STAGE, f"{tag}__ins-own__rep0"), keys).require_complete()
    return sum(bool(r["success"]) for r in rd.records.values()), len(rd.records), rd


def main():
    keys = grid(range(10), split_states("calibration"))
    targets = [float(t) for t in protocol("corruption.targets")]
    ext = protocol("corruption.ladder_extension")
    k0, n0, _ = load(CorruptionSpec("identity").tag, keys)
    clean = k0 / n0
    out = {"clean": {"success": k0, "n": n0, "wilson95": list(wilson_ci(k0, n0))}, "targets": targets, "families": {}}
    lines = ["# Severity calibration: success vs severity", "",
             f"Calibration split, 10 tasks x {len(split_states('calibration'))} init states, both cameras. "
             f"Clean: {k0}/{n0}.", ""]
    for fam in LADDER_KEY:
        levels = ladder(fam)
        ks, ns, cis = [], [], []
        for v in levels:
            k, n, _ = load(CorruptionSpec(fam, v).tag, keys)
            ks.append(k), ns.append(n), cis.append(list(wilson_ci(k, n)))
        sel = select(levels, ks, ns, clean, targets)
        extension = None
        if sel["top_above_lowest_target"]:
            nxt = next_level(levels, ext[fam]["spacing"], ext[fam]["bound"])
            extension = nxt if nxt is not None else "none possible (physical bound)"
            ext_runs = list(STAGE.glob(f"{CorruptionSpec(fam, nxt).tag}__ins-own__rep0")) if nxt is not None else []
            if ext_runs:  # the rule's one-step extension was run: select again on the extended ladder
                k, n, _ = load(CorruptionSpec(fam, nxt).tag, keys)
                levels = levels + [nxt]
                ks, ns, cis = ks + [k], ns + [n], cis + [list(wilson_ci(k, n))]
                sel = select(levels, ks, ns, clean, targets)
            elif nxt is not None:
                extension = f"{nxt:g} (due, not yet run)"
        sel |= {"success": ks, "n": ns, "wilson95": cis, "extension": extension}
        out["families"][fam] = sel
        lines += [f"## {fam} ({UNITS[fam]})", "", "| level | success | 95% CI | isotonic fit / clean |", "|---|---|---|---|"]
        for v, k, n, ci, r in zip(levels, ks, ns, cis, sel["relative"]):
            lines.append(f"| {v:g} | {k}/{n} | [{ci[0]:.0%}, {ci[1]:.0%}] | {r:.2f} |")
        lines += ["", "Nearest level per target: " + ", ".join(f"{t:.0%} -> {sel['chosen'][t]:g}" for t in targets),
                  f"Selected levels: {[float(v) for v in sel['selected_levels']]}"
                  + (f"; unreachable targets {sel['unreachable_targets']}" if sel["unreachable_targets"] else "")
                  + ("; BENIGN (no target reachable)" if sel["benign"] else "")]
        if sel["extension"] is not None:
            lines.append(f"Ladder extension (rule): {sel['extension']}")
        lines.append("")
    atomic_write_json(STAGE / "summary.json", out)
    atomic_write_bytes(STAGE / "summary.md", ("\n".join(lines) + "\n").encode())
    figure(out)
    print("\n".join(lines))


def figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c, blue = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
    fams = list(out["families"])
    fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 3.1), sharey=True)
    for ax, fam in zip(axes, fams):
        f = out["families"][fam]
        x = list(range(len(f["levels"])))
        rate = [k / n for k, n in zip(f["success"], f["n"])]
        lo = [r - c[0] for r, c in zip(rate, f["wilson95"])]
        hi = [c[1] - r for r, c in zip(rate, f["wilson95"])]
        clean = out["clean"]["success"] / out["clean"]["n"]
        for t in out["targets"]:
            ax.axhline(t * clean, color=grid_c, lw=1, zorder=0)
            ax.text(len(x) - 0.5, t * clean, f"{t:.0%}", color=muted, fontsize=7, va="bottom", ha="right")
        ax.plot(x, [v * clean for v in f["relative"]], color=blue, lw=2, zorder=2)
        ax.errorbar(x, rate, yerr=[lo, hi], fmt="o", ms=5, color=blue, mec="#fcfcfb", mew=1.5, ecolor=blue,
                    elinewidth=1, capsize=0, zorder=3)
        for t, v in f["chosen"].items():
            i = f["levels"].index(v)
            ax.plot(i, f["relative"][i] * clean, marker="o", ms=11, mfc="none", mec=ink, mew=1.2, zorder=4)
        ax.set_xticks(x, [f"{v:g}" for v in f["levels"]], fontsize=7, color=muted)
        ax.set_xlabel(UNITS[fam], fontsize=8, color=ink)
        ax.set_title(fam, fontsize=9, color=ink)
        ax.set_ylim(-0.03, 1.05)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid_c)
        ax.tick_params(colors=muted, labelsize=7)
    axes[0].set_ylabel("task success (50 episodes per level)", fontsize=8, color=ink)
    fig.suptitle("Calibration: dots = observed success with Wilson 95% CI, line = monotone fit, "
                 "rings = levels selected for 90% / 60% / 30% of clean success", fontsize=9, color=ink)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"success_vs_severity.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    main()
