"""Clean baseline: does our pipeline reproduce OFT's published LIBERO-Goal success, and what is the noise floor?

Usage:  python scripts/analyze_baseline.py   ->  results/baseline/summary.json + summary.md (and prints the table)

Reads the episodes of the two clean replicates (results/baseline/identity-*__rep{0,1}/episodes*.jsonl).
Reported:
  * success per replicate with Wilson 95% CI, per task; whether the published value (97.9%, Kim et al. 2025,
    500 episodes) lies inside our interval
  * replicate noise floor: outcome discordance (exact McNemar) and trajectory-level disagreement (steps)
  * every failure with the intent oracle's view (which task goals became true, object displacement)
  * errored episodes (excluded and counted; must be 0)
Analyses use the last error-free record per (task, init state).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.metrics.stats import mcnemar_exact, wilson_ci  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json  # noqa: E402

PUBLISHED = 0.979  # OpenVLA-OFT, LIBERO-Goal (Kim, Finn, Liang 2025, Table 1)
RESULTS = ROOT / "results" / "baseline"


def load(rep: int):
    """Strict ingestion (vlamech.analysis.ingest): all 500 keys, no conflicts, clean provenance, or refuse."""
    from vlamech.analysis.ingest import find_run, grid, load_run

    keys = grid(range(10), range(50))
    rd = load_run(find_run(RESULTS, f"identity-*__ins-own__rep{rep}"), keys).require_complete()
    return {"dir": rd.run_dir, "recs": rd.records, "errors": rd.errored_only, "accounting": rd.accounting()}


def main():
    reps = [load(0), load(1)]
    from vlamech.analysis.ingest import grid

    keys = sorted(grid(range(10), range(50)))  # explicit pairing: both replicates are complete (checked on load)
    out = {"published_success": PUBLISHED, "n_common": len(keys), "replicates": [],
           "accounting": [r["accounting"] for r in reps]}
    lines = ["# Clean baseline", "", f"Published OFT LIBERO-Goal success: {PUBLISHED:.1%} (500 episodes)", ""]
    for i, rep in enumerate(reps):
        r = rep["recs"]
        k = sum(r[x]["success"] for x in r)
        lo, hi = wilson_ci(k, len(r))
        per_task = {}
        for t in range(10):
            ks = [x for x in r if x[0] == t]
            per_task[t] = {"n": len(ks), "success": sum(r[x]["success"] for x in ks)}
        out["replicates"].append({"replicate": i, "dir": str(rep["dir"].relative_to(ROOT)), "n": len(r), "success": k,
                                  "rate": k / len(r), "wilson95": [lo, hi], "published_inside_ci": lo <= PUBLISHED <= hi,
                                  "errors_unresolved": rep["errors"], "per_task": per_task})
        lines.append(f"Replicate {i}: {k}/{len(r)} = {k/len(r):.1%}  95% CI [{lo:.1%}, {hi:.1%}]  "
                     f"published inside CI: {lo <= PUBLISHED <= hi}  unresolved errors: {len(rep['errors'])}")
    a = [reps[0]["recs"][x]["success"] for x in keys]
    b = [reps[1]["recs"][x]["success"] for x in keys]
    mc = mcnemar_exact(a, b)
    step_diff = [x for x in keys if reps[0]["recs"][x]["steps"] != reps[1]["recs"][x]["steps"]]
    k_disc = mc["n10"] + mc["n01"]
    disc_ci = wilson_ci(k_disc, len(keys))
    out["noise_floor"] = {"outcome_discordant": k_disc, "discordance_wilson95": list(disc_ci), "mcnemar": mc,
                          "trajectory_differs": [list(x) for x in step_diff],
                          "outcome_discordance_rate": (mc["n10"] + mc["n01"]) / len(keys)}
    lines += ["", f"Noise floor over {len(keys)} paired episodes: outcome discordant {k_disc} "
              f"({k_disc/len(keys):.2%}, Wilson 95% CI [{disc_ci[0]:.2%}, {disc_ci[1]:.2%}]: an upper bound on run-to-run "
              f"outcome noise, not proof of zero numerical variability); trajectory (step count) differs in {len(step_diff)}: "
              f"{[f'task{t}-init{s}' for t, s in step_diff]}", "", "| Task | Instruction | Rep 0 | Rep 1 |", "|---|---|---|---|"]
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    for t in range(10):
        p = [out["replicates"][i]["per_task"][t] for i in (0, 1)]
        lines.append(f"| {t} | {suite.get_task(t).language} | {p[0]['success']}/{p[0]['n']} | {p[1]['success']}/{p[1]['n']} |")
    lines += ["", "Failures (replicate 0) with the intent oracle:", "",
              "| Episode | steps | goals that became true | objects moved > 1 cm | max env force (N) |", "|---|---|---|---|---|"]
    fails = []
    for x in sorted(reps[0]["recs"]):
        r = reps[0]["recs"][x]
        if r["success"]:
            continue
        moved = {o: round(d, 3) for o, d in r["obj_max_disp"].items() if d > 0.01}
        fails.append({"task": x[0], "init": x[1], "steps": r["steps"], "goals_true": r["goal_first_step"], "moved_m": moved,
                      "max_force": r["max_contact_force"], "off_table": r["off_table"]})
        lines.append(f"| task{x[0]}-init{x[1]} | {r['steps']} | {r['goal_first_step'] or '-'} | {moved or '-'} | "
                     f"{r['max_contact_force']['environment']:.0f} |")
    lines.append("\n(The 1 cm column is descriptive only; the oracle's displacement threshold is set per task by the "
                 "protocol rule from clean calibration episodes.)")
    out["failures_rep0"] = fails
    atomic_write_json(RESULTS / "summary.json", out)
    atomic_write_bytes(RESULTS / "summary.md", ("\n".join(lines) + "\n").encode())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
