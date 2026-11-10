"""Oracle thresholds from clean episodes, by the protocol rules (oracle.*).

Usage:  python scripts/calibrate_oracle.py   ->  results/oracle_calibration/{thresholds.json, episodes.md}

Input: the clean calibration run (results/severity_calibration/identity-*__ins-own__rep0; traces) and the clean
baseline (both replicates, records). Rules (configs/protocol.yaml oracle.*), applied to SUCCESSFUL clean episodes:
  disp_m[task]     max displacement of any non-target movable object (outside the task's own entities), over that
                   task's clean baseline successes (about 100 per task)
  joint[j]         max |q_j - q_j(post-settle)| over the successful episodes of every task for which j is not a
                   goal joint
  force[task]      max robot-object and max robot-environment contact force over that task's clean baseline successes
  floors           passive drift of the SAME task and init state with no robot action
                   (results/oracle_calibration/passive_drift.json, scripts/measure_passive_drift.py); an object / joint
                   counts as moved only beyond max(threshold, floor) (classifier v2)
A task without a successful calibration episode has no threshold: the script refuses (nothing is imputed).
The thresholds are frozen into configs/frozen_levels.yaml together with the severity selection; this script
does not write configs. Every calibration episode is also classified with the resulting thresholds (episodes.md):
successes must classify as success by construction; failures are listed for video inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.ingest import find_run, grid, load_run, trace  # noqa: E402
from vlamech.config import protocol, split_states  # noqa: E402
from vlamech.envs.trace import TRACE_SCHEMA  # noqa: E402
from vlamech.metrics.outcome import CLASSIFIER_VERSION, Thresholds, classify  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json, file_sha256, read_json  # noqa: E402

STAGE = ROOT / "results" / "severity_calibration"
OUT = ROOT / "results" / "oracle_calibration"


def main():
    states = split_states("calibration")
    tasks = range(10)
    rd = load_run(find_run(STAGE, "identity-*__ins-own__rep0"), grid(tasks, states)).require_complete()
    traces = {k: trace(rd, k) for k in sorted(rd.records)}
    ok = {k for k, r in rd.records.items() if r["success"]}
    no_success = [t for t in tasks if not any(k[0] == t for k in ok)]
    if no_success:
        raise RuntimeError(f"tasks {no_success} have no successful calibration episode: no threshold by rule")

    any_tr = next(iter(traces.values()))
    joints = any_tr["joint_names"]
    disp_m, force, joint = {}, {}, {j: 0.0 for j in joints}
    for t in tasks:
        keys = sorted(k for k in ok if k[0] == t)
        d = 0.0
        for k in keys:
            tr = traces[k]
            tg = tr["targets"][t]
            idx = [tr["obj_names"].index(o) for o in tg["non_target_objects"]]
            if idx:
                d = max(d, float(np.linalg.norm(tr["obj_pos"][:, idx] - tr["obj_pos"][0, idx], axis=-1).max()))
            for j in tg["non_target_joints"]:
                q = tr["joint_q"][:, tr["joint_names"].index(j)]
                joint[j] = max(joint[j], float(np.abs(q - q[0]).max()))
        disp_m[t] = d  # calibration-only value, kept for comparison
        force[t] = {c: max(rd.records[k]["max_contact_force"][c] for k in keys) for c in ("object", "environment")}

    # Nominal maxima of object displacement and contact force from the clean baseline successes (2 replicates x 500
    # episodes, every init state): a maximum needs many samples, and 5 calibration episodes per task underestimate it.
    # Clean data only: no corrupted outcome is used.
    base = [load_run(find_run(ROOT / "results" / "baseline", f"identity-*__ins-own__rep{r}"),
                   grid(tasks, range(int(protocol("splits.n_init_states"))))).require_complete() for r in (0, 1)]
    any_tg = next(iter(traces.values()))["targets"]
    disp_base, force_base, n_base = {}, {}, {}
    for t in tasks:
        recs = [r for run in base for k, r in run.records.items() if k[0] == t and r["success"]]
        n_base[t] = len(recs)
        disp_base[t] = max(r["obj_max_disp"][o] for r in recs for o in any_tg[t]["non_target_objects"]) \
            if any_tg[t]["non_target_objects"] else 0.0
        force_base[t] = {c: max(r["max_contact_force"][c] for r in recs) for c in ("object", "environment")}
    disp_calib, force_calib = disp_m, force
    disp_m, force = disp_base, force_base

    # every joint must have been observed as a non-goal joint in some task (else its threshold is undefined)
    unobserved = [j for j in joints if not any(j in traces[k]["targets"][k[0]]["non_target_joints"] for k in ok)]
    if unobserved:
        raise RuntimeError(f"joints {unobserved} never observed as non-goal joints in successful episodes")

    drift_path = OUT / "passive_drift.json"
    if not drift_path.exists():
        raise FileNotFoundError(f"{drift_path} missing: run scripts/measure_passive_drift.py")
    floor = read_json(drift_path)["floor"]

    rows, counts = [], {}
    for k in sorted(traces):
        f = floor[f"{k[0]}-{k[1]}"]
        c = classify(traces[k], k[0], Thresholds(disp_m[k[0]], joint, f["objects"], f["joints"]))
        if rd.records[k]["success"] != (c["outcome"] == "success"):
            raise RuntimeError(f"{k}: classifier and env disagree on success")
        counts[c["outcome"]] = counts.get(c["outcome"], 0) + 1
        rows.append((k, rd.records[k]["steps"], c))

    out = {
        "classifier_version": CLASSIFIER_VERSION, "trace_schema": TRACE_SCHEMA,
        "source_run": str(rd.run_dir.relative_to(ROOT)), "manifest_digest": rd.manifest["digest"],
        "rules": {k: protocol(f"oracle.{k}") for k in ("displacement_threshold_rule", "joint_threshold_rule",
                                                       "force_threshold_rule")},
        "n_success": {t: sum(1 for k in ok if k[0] == t) for t in tasks}, "n_episodes": len(rd.records),
        "nominal_source": "clean baseline successes (both replicates) for disp_m and force; calibration traces for joints",
        "n_success_baseline": n_base, "disp_m_calibration_only": disp_calib, "force_calibration_only": force_calib,
        "disp_m": disp_m, "joint": joint, "force": force, "calibration_outcomes": counts,
        "passive_drift_sha256": file_sha256(drift_path),
    }
    atomic_write_json(OUT / "thresholds.json", out)
    lines = ["# Oracle calibration", "", f"Source: `{out['source_run']}` (manifest {out['manifest_digest'][:12]}), "
             f"classifier v{CLASSIFIER_VERSION}, trace schema {TRACE_SCHEMA}", "",
             "| Task | clean baseline successes | disp threshold (µm) | max object force (N) | max env force (N) | env force, 5 calib eps |",
             "|---|---|---|---|---|---|"]
    for t in tasks:
        lines.append(f"| {t} | {n_base[t]} | {disp_m[t] * 1e6:.4g} | {force[t]['object']:.1f} | "
                     f"{force[t]['environment']:.1f} | {force_calib[t]['environment']:.1f} |")
    lines += ["", "| Joint | threshold |", "|---|---|"] + [f"| {j} | {v:.3g} |" for j, v in joint.items()]
    lines += ["", "Floors: passive drift of each episode's own task and init state (passive_drift.json)."]
    lines += ["", f"Calibration episodes by outcome: {counts}", "", "| Episode | steps | outcome | subtype | events |",
              "|---|---|---|---|---|"]
    for k, steps, c in rows:
        if c["outcome"] != "success":
            ev = c["events"]
            lines.append(f"| task{k[0]}-init{k[1]} | {steps} | {c['outcome']} | {c['subtype'] or '-'} | "
                         f"onsets {ev['goal_onset']}, progress {ev['target_progress']}, off-target {ev['off_target_first']} "
                         f"{ {k: f'{v:.3g}' for k, v in ev['off_target_size'].items()} } |")
    atomic_write_bytes(OUT / "episodes.md", ("\n".join(lines) + "\n").encode())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
