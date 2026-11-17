"""When sensors disagree: analysis of scripts/run/sensor_conflict.py and the sensor_faults runs (protocol sensor_conflict).

Usage:  python scripts/analysis/sensor_conflict.py [--partial]
  -> results/sensor_conflict/summary.{json,md}, results/figures/sensor_conflict/*.png

Open loop (held-out files in results/sensor_conflict/confirmatory/, probe trained on exploratory/):
  shares        per edit and phase, the |Delta|^2-weighted Shapley share of each source (sensor_conflict.share)
  tests A-F     as pre-specified (sensor_conflict.tests): stratified episode bootstrap (tasks fixed), bootstrap p,
                Holm within each family
  descriptive   stale-edit shares, off-axis movement under single-source conflicts, gripper decisions, render noise
                floor, patching curves (removed share by depth; 2000 bootstrap draws per point)
Closed loop (results/sensor_faults/): success and unsafe outcome (collision above the task's clean maximum or an
  object off the table, the frozen hazard proxies) against the paired 'none' run: exact McNemar, Holm within each
  table, paired differences with stratified cluster-bootstrap intervals.
--partial reports whatever has finished (for monitoring); the reported analysis uses complete runs only.
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

from vlamech.analysis import arbitration as A  # noqa: E402
from vlamech.config import frozen, protocol, split_states  # noqa: E402
from vlamech.metrics.stats import auroc, hierarchical_bootstrap, holm, mcnemar_exact, wilson_ci  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, atomic_write_json, file_sha256, read_json  # noqa: E402

OUT = ROOT / "results" / "sensor_conflict"
FAULTS_DIR = ROOT / "results" / "sensor_faults"
FIG = ROOT / "results" / "figures" / "sensor_conflict"
PHASES = ("approach", "pregrasp", "transport", "place")
EDITS = {"object": ("agentview", "wrist"), "arm": ("agentview", "wrist", "proprio"),
         "gripper": ("agentview", "wrist", "proprio"), "stale": ("agentview", "wrist", "proprio"),
         "arm-blur-8": ("agentview", "wrist", "proprio"), "arm-noise-128": ("agentview", "wrist", "proprio")}
PROBE_EDITS = ("object", "arm", "gripper")
PROBE_LAYERS = (8, 16, 24)
CONFLICTS = {"agentview": ("object", 1), "wrist": ("object", 2), "proprio": ("arm", 4)}  # conflict -> (edit, input)
PATCH_GROUPS = ("source", "instruction", "readout")
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
COLORS = {"agentview": "#2a78d6", "wrist": "#eb6834", "proprio": "#1baf7a", "instruction": "#8a5cd1", "readout": INK}
N_BOOT = int(protocol("statistics.n_boot"))


def boot(df, stat, null=None, n_boot=N_BOOT):
    return hierarchical_bootstrap(df, stat, levels=("init_state",), strata=("task",), n_boot=n_boot, null=null)


def episodes(split: str):
    for f in sorted((OUT / split).glob("task*-init*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if "skipped" in meta:
            continue
        yield meta, z


# --------------------------------------------------------------------------------------------------- open loop
def share_rows(split: str) -> pd.DataFrame:
    rows = []
    for meta, z in episodes(split):
        for ph in PHASES:
            for edit, src in EDITS.items():
                k = f"{ph}__{edit}__actions"
                if k not in z.files:
                    continue
                acts = z[k]
                v, delta = A.values(acts)
                phi = A.shapley(v, len(src))
                row = {"task": meta["task"], "init_state": int(meta["key"].split("init")[1]), "phase": ph, "edit": edit,
                       "delta2": float(delta @ delta), "grip": A.gripper_closed(acts).astype(int).tolist()}
                for m, s in enumerate(src):
                    row[f"phi_{s}"] = float(phi[m])
                    one = A.single_source(acts, m)
                    row[f"v_{s}"], row[f"resid2_{s}"] = one["v"], one["residual2"]
                if edit == "object" and f"{ph}__distractor__actions" in z.files:
                    d = z[f"{ph}__distractor__actions"]
                    dd = (d[1] - d[0])[:, A.ARM].ravel()
                    row["distractor_delta2"] = float(dd @ dd)
                    w = meta["worlds"][ph]
                    row["px_object"], row["px_distractor"] = w["object"]["changed_px"], w["distractor"]["changed_px"]
                r = z[f"{ph}__render__actions"]
                rr = (r[0] - r[1])[:, A.ARM].ravel()
                row["render2"] = float(rr @ rr)
                rows.append(row)
    return pd.DataFrame(rows)


def wshare(d: pd.DataFrame, s: str) -> float:
    w = d["delta2"].to_numpy()
    return float((d[f"phi_{s}"].to_numpy() * w).sum() / w.sum())


def shares_table(df: pd.DataFrame) -> dict:
    out = {}
    for edit, src in EDITS.items():
        for ph in PHASES:
            d = df[(df.edit == edit) & (df.phase == ph)]
            if d.empty:
                continue
            e = out.setdefault(edit, {})[ph] = {"n": int(len(d)), "median_delta": float(np.sqrt(d.delta2).median()),
                                                "render_noise_ratio": float(np.sqrt(d.render2.sum() / d.delta2.sum()))}
            for s in src:
                b = boot(d, lambda x, s=s: wshare(x, s), n_boot=2000)
                e[s] = [b["estimate"], b["ci_low"], b["ci_high"]]
                e[f"off_axis_{s}"] = float(np.sqrt(d[f"resid2_{s}"].sum() / d.delta2.sum()))
    return out


def tests(df: pd.DataFrame) -> dict:
    fam = {}

    def run(name, items):
        res = []
        for label, d, stat, null in items:
            if len(d) < 5:
                continue
            b = boot(d, stat, null=null)
            res.append({"label": label, "n": int(len(d)), "estimate": b["estimate"], "ci": [b["ci_low"], b["ci_high"]],
                        "p_boot": b["p_boot"]})
        if res:
            for r, p in zip(res, holm([r["p_boot"] for r in res])):
                r["p_holm"] = float(p)
        fam[name] = res

    sel = lambda e, ph: df[(df.edit == e) & (df.phase == ph)]  # noqa: E731
    run("A_wrist_minus_agentview_object", [(ph, sel("object", ph), lambda x: wshare(x, "wrist") - wshare(x, "agentview"), 0.0)
                                           for ph in PHASES])
    run("B_proprio_share_arm", [(ph, sel("arm", ph), lambda x: wshare(x, "proprio"), 0.0) for ph in PHASES])
    run("C_proprio_share_gripper", [(ph, sel("gripper", ph), lambda x: wshare(x, "proprio"), 0.0) for ph in PHASES])
    rel = df[(df.edit == "object") & df.distractor_delta2.notna() & (df.distractor_delta2 > 0)].copy() \
        if "distractor_delta2" in df else df.iloc[:0]
    if len(rel):
        rel["log_ratio"] = 0.5 * np.log(rel.delta2 / rel.distractor_delta2)
    run("D_relevance_log_ratio", [(ph, rel[rel.phase == ph], lambda x: float(x.log_ratio.mean()), 0.0) for ph in PHASES])
    items = []
    clean = df[df.edit == "arm"].set_index(["task", "init_state", "phase"])
    for c in ("arm-blur-8", "arm-noise-128"):
        cor = df[df.edit == c].set_index(["task", "init_state", "phase"])
        both = clean.join(cor, lsuffix="_c", rsuffix="_x", how="inner").reset_index()

        def diff(x):
            wc, wx = x.delta2_c.to_numpy(), x.delta2_x.to_numpy()
            return float((x.phi_proprio_x * wx).sum() / wx.sum() - (x.phi_proprio_c * wc).sum() / wc.sum())

        items.append((c, both, diff, 0.0))
    run("E_proprio_share_corrupted_minus_clean", items)
    return fam


def gripper_decisions(df: pd.DataFrame) -> dict:
    """Gripper edit: share of states where the chunk's close/open decision follows the proprio rather than the images."""
    out = {}
    for ph in PHASES:
        d = df[(df.edit == "gripper") & (df.phase == ph)]
        if d.empty:
            continue
        g = np.array(d.grip.tolist())  # (n, 8): input index = agentview + 2 wrist + 4 proprio
        real, edited = g[:, 0], g[:, 7]
        flips = real != edited
        out[ph] = {"n": int(len(d)), "edited_world_flips_decision": float(flips.mean()),
                   "proprio_only_flips_decision": float((g[:, 4] != real).mean()),
                   "cameras_only_flip_decision": float((g[:, 3] != real).mean())}
    return out


# ------------------------------------------------------------------------------------------------------- probe
def probe_data(split: str, layer: int):
    X, y, groups, edits = [], [], [], []
    li = PROBE_LAYERS.index(layer)
    for meta, z in episodes(split):
        for ph in PHASES:
            for edit in PROBE_EDITS:
                k = f"{ph}__{edit}__feats"
                if k not in z.files:
                    continue
                f = z[k][:, li].astype(np.float32)
                n = f.shape[0]
                lab = np.ones(n, int)
                lab[[0, n - 1]] = 0
                X.append(f)
                y.append(lab)
                groups += [(meta["task"], int(meta["key"].split("init")[1]))] * n
                edits += [edit] * n
    return np.concatenate(X), np.concatenate(y), groups, np.array(edits)


def probe() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    for layer in PROBE_LAYERS:
        Xtr, ytr, _, etr = probe_data("exploratory", layer)
        Xte, yte, gte, ete = probe_data("confirmatory", layer)
        if len(Xtr) == 0 or len(Xte) == 0:
            return out
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xtr, ytr)
        s = clf.decision_function(Xte)
        df = pd.DataFrame({"task": [g[0] for g in gte], "init_state": [g[1] for g in gte], "s": s, "y": yte, "edit": ete})
        b = boot(df, lambda d: auroc(d.s.to_numpy(), d.y.to_numpy()), null=0.5, n_boot=2000 if layer != 16 else N_BOOT)
        res = {"auroc": [b["estimate"], b["ci_low"], b["ci_high"]], "p_boot": b["p_boot"], "n_test": int(len(df))}
        res["by_edit"] = {e: auroc(df[df.edit == e].s.to_numpy(), df[df.edit == e].y.to_numpy()) for e in PROBE_EDITS}
        # generalization to an edit the probe never saw
        res["held_out_edit"] = {}
        for e in PROBE_EDITS:
            c2 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xtr[etr != e], ytr[etr != e])
            m = ete == e
            res["held_out_edit"][e] = auroc(c2.decision_function(Xte[m]), yte[m])
        out[str(layer)] = res
    return out


# ---------------------------------------------------------------------------------------------------- patching
def patching() -> dict:
    grid = [str(p) for p in protocol("sensor_conflict.patch_grid")]
    rows = []
    for meta, z in episodes("confirmatory"):
        for ph in ("pregrasp", "transport"):
            for conflict, (edit, idx) in CONFLICTS.items():
                k = f"{ph}__patch__{conflict}"
                if k not in z.files or f"{ph}__{edit}__actions" not in z.files:
                    continue
                acts = z[f"{ph}__{edit}__actions"]
                a0, at = acts[0][:, A.ARM].ravel(), acts[idx][:, A.ARM].ravel()
                s = at - a0
                for gi, g in enumerate(PATCH_GROUPS):
                    for pi, p in enumerate(grid):
                        ap = z[k][gi, pi][:, A.ARM].ravel()
                        rows.append({"task": meta["task"], "init_state": int(meta["key"].split("init")[1]), "phase": ph,
                                     "conflict": conflict, "group": g, "point": p, "num": float((ap - a0) @ s),
                                     "s2": float(s @ s)})
    df = pd.DataFrame(rows)
    out = {"grid": grid}
    if df.empty:
        return out
    for (ph, c, g), d in df.groupby(["phase", "conflict", "group"]):
        curve = []
        for p in grid:
            dp = d[d.point == p]
            b = boot(dp, lambda x: 1 - x.num.sum() / x.s2.sum(), n_boot=2000)
            curve.append([b["estimate"], b["ci_low"], b["ci_high"]])
        out.setdefault(ph, {}).setdefault(c, {})[g] = curve
        out[ph][c]["n"] = int(d[["task", "init_state"]].drop_duplicates().shape[0])
    return out


# ------------------------------------------------------------------------------------------------- closed loop
QUICK = {"dir": ROOT / "results" / "debug", "states": range(20, 30),
         "faults": ("none", "delay_2", "delay_4", "delay_8", "delay_16", "lag_agentview", "lag_wrist", "lag_proprio",
                    "lag_all", "proprio_offset", "gripper_stuck", "freeze_transport", "rescue_wrist_12", "rescue_readout_20",
                    "rescue_wrist_12_random", "rescue_readout_20_random")}
RESCUES = QUICK["faults"][-4:]
DELAY_S = {"none": 0.0, "delay_2": 0.1, "delay_4": 0.2, "delay_8": 0.4, "delay_16": 0.8}


def behaviour(tr: dict) -> dict:
    """Hand and gripper behaviour from a trace: mean hand speed (mm/step), share of 8-step hand displacements that reverse
    the previous one (cos < 0), and grasp cycles (finger opening falling below 4 cm)."""
    eef = np.asarray(tr["eef_pos"])
    v = np.linalg.norm(np.diff(eef, axis=0), axis=1) * 1000
    d = eef[8::8] - eef[:-8:8] if len(eef) > 16 else np.zeros((0, 3))
    n = np.linalg.norm(d, axis=1)
    ok = (n[1:] > 2e-3) & (n[:-1] > 2e-3)
    cos = (d[1:] * d[:-1]).sum(1)[ok] / (n[1:] * n[:-1])[ok] if ok.any() else np.array([])
    op = np.asarray(tr["gripper_q"])
    opening = op[:, 0] - op[:, 1]
    closed = opening < 0.04
    return {"speed": float(v.mean()) if len(v) else 0.0, "reversal": float((cos < 0).mean()) if len(cos) else 0.0,
            "grasp_cycles": int((closed[1:] & ~closed[:-1]).sum())}


def closed_loop(partial: bool, quick: bool = False) -> dict:
    from vlamech.analysis.ingest import grid as keygrid, load_run, trace
    from vlamech.interventions.sensor_faults import FAULTS, PHASE_FAULTS

    thr_path = ROOT / "results" / "oracle_calibration" / "thresholds.json"
    if file_sha256(thr_path) != frozen("oracle.thresholds_sha256"):
        raise RuntimeError("oracle thresholds differ from the frozen ones")
    thr = read_json(thr_path)
    states = split_states("confirmatory")
    pick = protocol("sensor_conflict.tasks")
    runs, missing = {}, []
    for f in (QUICK["faults"] if quick else FAULTS + PHASE_FAULTS):
        d = QUICK["dir"] / f"quick-fault-{f}" if quick else FAULTS_DIR / f"identity-agentview+wrist__ins-own__fault-{f}__rep0"
        keys = keygrid(pick if quick or f not in FAULTS else range(10), QUICK["states"] if quick else states)
        try:
            rd = load_run(d, keys)
        except FileNotFoundError:
            missing.append(f); continue
        clean_but_dirty = quick and not (rd.missing or rd.errored_only or rd.conflicts or rd.unexpected or rd.foreign_stamps)
        if not rd.complete and not clean_but_dirty:  # quick runs are debug runs: an uncommitted tree is allowed
            if not partial:
                rd.require_complete()
            missing.append(f)  # partial: paired on the episodes finished so far
        runs[f] = rd
    if "none" not in runs:
        return {"missing": missing}
    rows = []
    for f, rd in runs.items():
        for k, r in rd.records.items():
            t = k[0]
            col = r["max_contact_force"]["environment"] > thr["force"][str(t)]["environment"]
            rows.append({"fault": f, "task": t, "init_state": k[1], "success": bool(r["success"]),
                         "unsafe": bool(col or r["off_table"]), "steps": r["steps"], **behaviour(trace(rd, k))})
    df = pd.DataFrame(rows)
    out = {"missing": missing, "faults": {}}
    tables = ((("faults", QUICK["faults"][1:], pick),) if quick else
              (("faults", FAULTS[1:], range(10)), ("phase_faults", PHASE_FAULTS, pick)))
    for table, names, tasks in tables:
        res = {}
        base = df[(df.fault == "none") & df.task.isin(list(tasks))].set_index(["task", "init_state"])
        res["none"] = {"success": float(base.success.mean()), "unsafe": float(base.unsafe.mean()), "n": int(len(base)),
                       "success_wilson": list(wilson_ci(int(base.success.sum()), len(base)))}
        ps = {"success": [], "unsafe": []}
        for f in names:
            if f not in runs:
                continue
            x = df[df.fault == f].set_index(["task", "init_state"])
            j = base.join(x, lsuffix="_0", rsuffix="_1", how="inner").reset_index()
            e = res[f] = {"n": int(len(j)), "success": float(j.success_1.mean()), "unsafe": float(j.unsafe_1.mean()),
                          "success_wilson": list(wilson_ci(int(j.success_1.sum()), len(j)))}
            for m in ("speed", "reversal", "grasp_cycles"):
                e[m], e[f"{m}_plain"] = float(j[f"{m}_1"].mean()), float(j[f"{m}_0"].mean())
            for m in ("success", "unsafe"):
                mc = mcnemar_exact(j[f"{m}_0"].to_numpy(), j[f"{m}_1"].to_numpy())
                b = boot(j, lambda d, m=m: float(d[f"{m}_1"].mean() - d[f"{m}_0"].mean()))
                e[f"{m}_diff"] = [b["estimate"], b["ci_low"], b["ci_high"]]
                e[f"{m}_mcnemar_p"] = mc["p"]
                ps[m].append((f, mc["p"]))
        for m, lst in ps.items():
            if lst:
                for (f, _), p in zip(lst, holm([p for _, p in lst])):
                    res[f][f"{m}_p_holm"] = float(p)
        out[table] = res
    if quick and "lag_wrist" in runs:  # rescue arms against the unpatched delayed wrist camera, paired
        base = df[df.fault == "lag_wrist"].set_index(["task", "init_state"])
        res, ps = {}, []
        for f in RESCUES:
            if f not in runs:
                continue
            j = base.join(df[df.fault == f].set_index(["task", "init_state"]), lsuffix="_0", rsuffix="_1", how="inner").reset_index()
            mc = mcnemar_exact(j.success_0.to_numpy(), j.success_1.to_numpy())
            b = boot(j, lambda d: float(d.success_1.mean() - d.success_0.mean()))
            res[f] = {"n": int(len(j)), "success": float(j.success_1.mean()), "unsafe": float(j.unsafe_1.mean()),
                      "success_vs_delayed": [b["estimate"], b["ci_low"], b["ci_high"]], "p": mc["p"]}
            ps.append(mc["p"])
        for f, p in zip(res, holm(ps) if ps else []):
            res[f]["p_holm"] = float(p)
        fresh = {}
        for f in ("rescue_wrist_12", "rescue_readout_20"):
            if f in runs and f"{f}_random" in runs:
                x = df[df.fault == f"{f}_random"].set_index(["task", "init_state"])
                j = x.join(df[df.fault == f].set_index(["task", "init_state"]), lsuffix="_0", rsuffix="_1", how="inner").reset_index()
                mc = mcnemar_exact(j.success_0.to_numpy(), j.success_1.to_numpy())
                b = boot(j, lambda d: float(d.success_1.mean() - d.success_0.mean()))
                fresh[f] = {"diff": [b["estimate"], b["ci_low"], b["ci_high"]], "p": mc["p"]}
        for f, p in zip(fresh, holm([v["p"] for v in fresh.values()]) if fresh else []):
            fresh[f]["p_holm"] = float(p)
        out["rescue"] = {"delayed_success": float(base.success.mean()), "arms": res, "fresh_vs_random": fresh}
    return out


def patching_stale() -> dict:
    """Removed share of the 0.4 s single-source delay conflicts' effect, by group and depth (descriptive)."""
    grid = [str(p) for p in protocol("sensor_conflict.patch_grid")]
    rows = []
    for f in sorted((OUT / "patching_stale").glob("task*-init*.npz")):
        z = np.load(f)
        t, i = int(f.stem[4:].split("-")[0]), int(f.stem.split("init")[1])
        for ph in ("pregrasp", "transport"):
            now = z[f"{ph}__now"][:, A.ARM].ravel()
            for src in ("agentview", "wrist", "proprio"):
                sv = z[f"{ph}__mixed__{src}"][:, A.ARM].ravel() - now
                for gi, g in enumerate(PATCH_GROUPS):
                    for pi, p in enumerate(grid):
                        ap = z[f"{ph}__patch__{src}"][gi, pi][:, A.ARM].ravel()
                        rows.append({"task": t, "init_state": i, "phase": ph, "conflict": src, "group": g, "point": p,
                                     "num": float((ap - now) @ sv), "s2": float(sv @ sv)})
    df = pd.DataFrame(rows)
    out = {"grid": grid}
    if df.empty:
        return out
    for (ph, c, g), d in df.groupby(["phase", "conflict", "group"]):
        curve = []
        for p in grid:
            b = boot(d[d.point == p], lambda x: 1 - x.num.sum() / x.s2.sum(), n_boot=1000)
            curve.append([b["estimate"], b["ci_low"], b["ci_high"]])
        out.setdefault(ph, {}).setdefault(c, {})[g] = curve
        out[ph][c]["n"] = int(d[["task", "init_state"]].drop_duplicates().shape[0])
        out[ph][c]["effect"] = float(np.sqrt(d[d.point == grid[0]].s2.mean()))
    return out


# ----------------------------------------------------------------------------------------------------- report
def fmt(x, d=2):
    return f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]"


def report(s: dict) -> str:
    L = ["# When sensors disagree (exploratory, reduced run)", "",
         "Open loop: 59 held-out episodes (every third successful pick-and-place episode, init states 20-49). Closed loop: "
         "pick-and-place tasks, init states 20-29, 60 paired episodes per fault. Patching was not run. Shares are the part of the policy's response to the edited "
         "world that follows each source (Shapley values, |Delta|^2-weighted; 95% stratified bootstrap CIs). Protocol: "
         "`configs/protocol.yaml` sensor_conflict, design: `docs/sensor_conflict.md`.", ""]
    sh = s.get("shares", {})
    for edit, src in EDITS.items():
        if edit not in sh:
            continue
        L += [f"## {edit}", "", "| phase | n | " + " | ".join(src) + " | median |Delta| | render noise / |Delta| |",
              "|---|---|" + "---|" * len(src) + "---|---|"]
        for ph, e in sh[edit].items():
            L.append(f"| {ph} | {e['n']} | " + " | ".join(fmt(e[x]) for x in src) +
                     f" | {e['median_delta']:.2f} | {e['render_noise_ratio']:.2f} |")
        L.append("")
    L += ["## Pre-specified tests", ""]
    for name, res in s.get("tests", {}).items():
        L += [f"**{name}**", "", "| | n | estimate [95% CI] | p (bootstrap) | p (Holm) |", "|---|---|---|---|---|"]
        for r in res:
            L.append(f"| {r['label']} | {r['n']} | {r['estimate']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | "
                     f"{r['p_boot']:.3g} | {r.get('p_holm', float('nan')):.3g} |")
        L.append("")
    if s.get("probe"):
        L += ["## Detecting the conflict from the policy's own state (probe)", "",
              "| layer | AUROC [95% CI] | p | object | arm | gripper | unseen edit: object / arm / gripper |", "|---|---|---|---|---|---|---|"]
        for layer, r in s["probe"].items():
            be, ho = r["by_edit"], r["held_out_edit"]
            L.append(f"| {layer} | {fmt(r['auroc'])} | {r['p_boot']:.3g} | {be['object']:.2f} | {be['arm']:.2f} | "
                     f"{be['gripper']:.2f} | {ho['object']:.2f} / {ho['arm']:.2f} / {ho['gripper']:.2f} |")
        L.append("")
    if s.get("gripper_decisions"):
        L += ["## Gripper decisions under the gripper edit", "", "| phase | n | edited world flips | proprio alone flips | cameras alone flip |",
              "|---|---|---|---|---|"]
        for ph, g in s["gripper_decisions"].items():
            L.append(f"| {ph} | {g['n']} | {g['edited_world_flips_decision']:.2f} | {g['proprio_only_flips_decision']:.2f} | "
                     f"{g['cameras_only_flip_decision']:.2f} |")
        L.append("")
    cl = s.get("closed_loop", {})
    for table in ("faults", "phase_faults"):
        if table not in cl:
            continue
        r = cl[table]
        L += [f"## Closed loop: {table.replace('_', ' ')}", "",
              f"Paired plain run: success {r['none']['success']:.3f}, unsafe {r['none']['unsafe']:.3f} (n = {r['none']['n']}).", "",
              "| fault | n | success | difference [95% CI] | p (Holm) | unsafe | difference [95% CI] | p (Holm) | hand speed mm/step (plain) | reversals (plain) | grasp cycles (plain) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for f, e in r.items():
            if f == "none":
                continue
            L.append(f"| {f} | {e['n']} | {e['success']:.3f} | {fmt(e['success_diff'], 3)} | {e.get('success_p_holm', float('nan')):.3g} | "
                     f"{e['unsafe']:.3f} | {fmt(e['unsafe_diff'], 3)} | {e.get('unsafe_p_holm', float('nan')):.3g} | "
                     f"{e['speed']:.1f} ({e['speed_plain']:.1f}) | {e['reversal']:.2f} ({e['reversal_plain']:.2f}) | "
                     f"{e['grasp_cycles']:.2f} ({e['grasp_cycles_plain']:.2f}) |")
        L.append("")
    if cl.get("rescue"):
        r = cl["rescue"]
        L += ["## Patching rescue of a 0.4 s late wrist camera (closed loop)", "",
              f"Unpatched delayed wrist camera: success {r['delayed_success']:.3f}; plain run {cl['faults']['none']['success']:.3f}.", "",
              "| arm | n | success | vs delayed [95% CI] | p (Holm) | unsafe |", "|---|---|---|---|---|---|"]
        for f, e in r["arms"].items():
            L.append(f"| {f} | {e['n']} | {e['success']:.3f} | {fmt(e['success_vs_delayed'], 3)} | {e.get('p_holm', float('nan')):.3g} | {e['unsafe']:.3f} |")
        L.append("")
        for f, e in r.get("fresh_vs_random", {}).items():
            L.append(f"Fresh states against the random donor, {f}: {fmt(e['diff'], 3)}, p (Holm) {e['p_holm']:.3g}.")
        L.append("")
    if cl.get("missing"):
        L.append(f"Closed-loop runs not complete: {', '.join(cl['missing'])}.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures(s: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    sh = s.get("shares", {})
    panels = [e for e in ("object", "arm", "gripper", "stale") if e in sh]
    if panels:
        titles = {"object": "object or destination moved 5 cm", "arm": "hand moved 3 cm",
                  "gripper": "gripper shown open / closed", "stale": "observation 0.4 s old"}
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6), facecolor=SURF, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, edit in zip(axes, panels):
            _style(ax)
            src = EDITS[edit]
            phs = [p for p in PHASES if p in sh[edit]]
            x = np.arange(len(phs))
            w = 0.8 / len(src)
            for i, so in enumerate(src):
                est = np.array([sh[edit][p][so] for p in phs])
                ax.bar(x + (i - (len(src) - 1) / 2) * w, est[:, 0], w * 0.92, color=COLORS[so], label=so,
                       edgecolor=SURF, linewidth=1)
                ax.errorbar(x + (i - (len(src) - 1) / 2) * w, est[:, 0], yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]],
                            fmt="none", ecolor=INK, elinewidth=0.8, capsize=0)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(x, phs, fontsize=8)
            ax.set_title(titles[edit], fontsize=9, color=INK)
        axes[0].set_ylabel("share of the response that follows the source", fontsize=8, color=INK)
        h, lab = axes[-1].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        fig.savefig(FIG / "shares.png", dpi=200, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
    for name, pt in (("patching", s.get("patching", {})), ("patching_delay", s.get("patching_stale", {}))):
        if "pregrasp" in pt:
            _patch_fig(plt, pt, name)
    cl = s.get("closed_loop", {})
    if "faults" in cl:
        _closed_fig(plt, cl)


def _patch_fig(plt, pt, name):
    if True:
        grid = pt["grid"]
        xs = [-1 if p == "embed" else int(p) for p in grid]
        conflicts = [c for c in CONFLICTS if c in pt["pregrasp"]]
        fig, axes = plt.subplots(2, len(conflicts), figsize=(3.6 * len(conflicts), 5.6), facecolor=SURF, sharex=True, sharey=True)
        axes = np.atleast_2d(axes) if len(conflicts) > 1 else np.array(axes).reshape(2, 1)
        for r, ph in enumerate(("pregrasp", "transport")):
            for c_i, c in enumerate(conflicts):
                ax = axes[r, c_i]
                _style(ax)
                if c not in pt.get(ph, {}):
                    continue
                for g in PATCH_GROUPS:
                    cur = np.array(pt[ph][c][g])
                    col = COLORS[c] if g == "source" else COLORS[g]
                    lab = f"{c} positions" if g == "source" else f"{g} positions"
                    ax.plot(xs, cur[:, 0], color=col, lw=2, label=lab)
                    ax.fill_between(xs, cur[:, 1], cur[:, 2], color=col, alpha=0.15, lw=0)
                ax.set_ylim(-0.2, 1.1)
                ax.set_title(f"{c} {'0.4 s late' if name == 'patching_delay' else 'conflict'}, {ph} (n = {pt[ph][c]['n']})",
                             fontsize=9, color=INK)
                if r == 1:
                    ax.set_xlabel("layer patched (-1 = input)", fontsize=8, color=INK)
                if c_i == 0:
                    ax.set_ylabel("share of the conflict's effect removed", fontsize=8, color=INK)
                ax.legend(frameon=False, fontsize=7, loc="center left")
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)


def _ladder_fig(plt, r):
    xs = [x for f, x in DELAY_S.items() if f in r]
    fs = [f for f in DELAY_S if f in r]
    if len(fs) < 3:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4), facecolor=SURF)
    _style(ax)
    for m, col, mk, lab in (("success", COLORS["agentview"], "o", "success"), ("unsafe", COLORS["wrist"], "s", "unsafe outcome")):
        y = [r[f][m] for f in fs]
        ax.plot(xs, y, color=col, lw=2, marker=mk, ms=6, mec=SURF, label=lab)
        if m == "success":
            lo = [r[f]["success_wilson"][0] for f in fs]
            hi = [r[f]["success_wilson"][1] for f in fs]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
    ax.set_xticks(xs, [f"{x:g}" for x in xs])
    ax.set_xlabel("observation delay (s), whole observation", fontsize=8, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"delay ladder, pick-and-place (n = {r[fs[1]]['n']} paired)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "delay_ladder.png", dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def _rescue_fig(plt, cl):
    r = cl["rescue"]["arms"]
    labels = ["plain", "wrist 0.4 s late"] + [f.replace("rescue_", "").replace("_random", " (random donor)").replace("_", " @ L")
                                                for f in r]
    vals = [cl["faults"]["none"]["success"], cl["rescue"]["delayed_success"]] + [e["success"] for e in r.values()]
    cols = [MUTED, COLORS["wrist"]] + [COLORS["agentview"] if not f.endswith("random") else GRID for f in r]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURF)
    _style(ax)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.7, edgecolor=SURF)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("task success (60 paired pick-and-place episodes)", fontsize=8, color=INK)
    ax.set_title("restoring one layer's states with the fresh wrist frame", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "rescue.png", dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def _closed_fig(plt, cl):
    _ladder_fig(plt, cl["faults"])
    if cl.get("rescue", {}).get("arms"):
        _rescue_fig(plt, cl)
    if True:
        tabs = [t for t in ("faults", "phase_faults") if t in cl]
        fig, axes = plt.subplots(1, len(tabs), figsize=(5.2 * len(tabs), 3.6), facecolor=SURF)
        axes = np.atleast_1d(axes)
        for ax, t in zip(axes, tabs):
            _style(ax)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, lw=0.6)
            r = cl[t]
            names = [f for f in r if f != "none" and not f.startswith("rescue")]
            y = np.arange(len(names))[::-1]
            for yi, f in zip(y, names):
                e = r[f]["success_diff"]
                ax.plot([e[1], e[2]], [yi, yi], color=COLORS["agentview"], lw=2, solid_capstyle="round")
                ax.plot(e[0], yi, "o", color=COLORS["agentview"], ms=6, mec=SURF, mew=1.2)
                u = r[f]["unsafe_diff"]
                ax.plot([u[1], u[2]], [yi - 0.25, yi - 0.25], color=COLORS["wrist"], lw=2, solid_capstyle="round")
                ax.plot(u[0], yi - 0.25, "s", color=COLORS["wrist"], ms=5, mec=SURF, mew=1.2)
            ax.axvline(0, color=MUTED, lw=0.8)
            ax.set_yticks(y - 0.12, names, fontsize=8)
            ax.set_xlabel("change against the paired plain run (95% CI)", fontsize=8, color=INK)
            ax.set_title("sensor faults, all tasks" if t == "faults" else "0.8 s freeze by phase, pick-and-place tasks",
                         fontsize=9, color=INK)
        axes[0].plot([], [], "o", color=COLORS["agentview"], label="success")
        axes[0].plot([], [], "s", color=COLORS["wrist"], label="unsafe outcome")
        axes[0].legend(frameon=False, fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(FIG / "closed_loop.png", dpi=200, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="exploratory: closed-loop faults from results/debug/quick-fault-* (pick-and-place tasks, init 20-29)")
    a = ap.parse_args()
    s = {}
    df = share_rows("confirmatory")
    if not df.empty:
        s["n_episodes"] = int(df[["task", "init_state"]].drop_duplicates().shape[0])
        s["shares"] = shares_table(df)
        s["tests"] = tests(df)
        s["gripper_decisions"] = gripper_decisions(df)
        s["probe"] = probe()
        if s["probe"].get("16"):
            p = s["probe"]["16"]
            s["tests"]["F_probe_auroc_layer16"] = [{"label": "layer 16", "n": p["n_test"], "estimate": p["auroc"][0],
                                                    "ci": p["auroc"][1:], "p_boot": p["p_boot"], "p_holm": p["p_boot"]}]
        s["patching"] = patching()
        s["patching_stale"] = patching_stale()
    s["closed_loop"] = closed_loop(a.partial, a.quick)
    atomic_write_json(OUT / "summary.json", s)
    atomic_write_bytes(OUT / "summary.md", report(s).encode())
    figures(s)
    print(report(s))


if __name__ == "__main__":
    main()
