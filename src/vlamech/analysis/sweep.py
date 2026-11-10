"""Corruption-sweep episode table: every frozen condition through the frozen oracle, plus exploratory labels.

Used by: scripts/analyze_corruption_sweep.py (confirmatory), scripts/figures_trajectories.py (exploratory figures).

build_table(root, partial) loads each condition with the strict ingestion (complete runs, clean code, manifest
stamps, trace hashes), checks the oracle thresholds against the frozen hash, and returns one row per episode:
oracle v2 outcome (confirmatory), the exploratory destination label (protocol oracle.destination_label), hazard
proxies as frozen (environment contact force above the task's clean baseline maximum; off table), plus per-step arrays for
the figures (hand path, speed, what is touched, collision steps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vlamech.analysis.ingest import grid, load_run, trace
from vlamech.config import frozen, split_states
from vlamech.metrics import destination as dst
from vlamech.metrics.outcome import Thresholds, classify
from vlamech.perturbations.image import CorruptionSpec
from vlamech.utils.io import file_sha256, read_json


def conditions() -> list:
    """Every frozen condition, clean first, then the exact-model-input variants."""
    sel = frozen()
    out = []
    for f in ["identity"] + sorted(f for f in sel["corruption"] if f != "identity"):
        for v in sel["corruption"][f]["levels"]:
            out.append(CorruptionSpec(f, v))
    for f in sel.get("variants", {}).get("exact_model_input", []):
        for v in sel["corruption"][f]["levels"]:
            out.append(CorruptionSpec(f, v, exact_model_input=True))
    return out


def label_of(spec: CorruptionSpec) -> str:
    return spec.tag.replace("-agentview+wrist", "")


def speeds(tr: dict) -> np.ndarray:
    return np.linalg.norm(np.diff(tr["eef_pos"], axis=0), axis=1) * 1000  # mm per control step


def touch_codes(tr: dict, task: int, env_thr: float) -> tuple:
    names = tr["entity_names"]
    own = set(tr["targets"][task]["touch_entities"])
    objs = set(tr["obj_names"])
    cat = []
    for n in names:
        cat.append(1 if n in own else 2 if n in objs else 4 if n == "main_table" else 3)
    c = np.asarray(tr["contact"])[1:]  # row 0 is the post-settle frame
    code = np.zeros(len(c), int)
    for j in np.argsort([-k for k in cat]):  # lower category number wins (own target over everything else)
        code[c[:, j] > 0] = cat[j]
    # collision: the frozen hazard proxy's own definition (LiberoTaskEnv.robot_contact_forces "environment" = every
    # entity except movable objects, own fixtures included) against the task's clean baseline maximum
    env = np.array([n not in objs for n in names])
    collide = (c[:, env] > env_thr).any(axis=1) if env.any() else np.zeros(len(c), bool)
    return code, collide


def build_table(root: Path, partial: bool):
    sweep = root / "results" / "corruption_sweep"
    keys = grid(range(10), split_states("confirmatory"))
    thr_path = root / "results" / "oracle_calibration" / "thresholds.json"
    if file_sha256(thr_path) != frozen("oracle.thresholds_sha256"):
        raise RuntimeError("oracle thresholds differ from the frozen ones (configs/frozen_levels.yaml)")
    thr = read_json(thr_path)
    floor = read_json(root / "results" / "oracle_calibration" / "passive_drift.json")["floor"]
    runs, skipped = {}, []
    for spec in conditions():
        d = sweep / f"{spec.tag}__ins-own__rep0"
        if not d.exists():
            if not partial:
                raise FileNotFoundError(d)
            skipped.append(label_of(spec)); continue
        try:
            rd = load_run(d, keys)
        except FileNotFoundError:  # run started, no episode written yet
            if not partial:
                raise
            skipped.append(label_of(spec)); continue
        if not rd.complete:
            if not partial:
                rd.require_complete()
            skipped.append(label_of(spec)); continue
        runs[label_of(spec)] = rd
    if "identity" not in runs:
        raise RuntimeError("the clean run is needed (destinations, twins)")
    clean = runs["identity"]
    clean_ok = [(k[0], trace(clean, k)) for k, r in sorted(clean.records.items()) if r["success"]]
    dest = dst.destinations(clean_ok)
    inc = dst.incidental(clean_ok, dest)
    rows, traces = [], {}
    for cond, rd in runs.items():
        for k in sorted(rd.records):
            r, tr = rd.records[k], trace(rd, k)
            t = k[0]
            f = floor[f"{t}-{k[1]}"]
            c = classify(tr, t, Thresholds(thr["disp_m"][str(t)], thr["joint"], f["objects"], f["joints"]))
            if (c["outcome"] == "success") != bool(r["success"]):
                raise RuntimeError(f"{cond} {k}: oracle and environment disagree on success")
            head = dst.heading(tr, t, dest, progressed=c["events"]["target_progress"] is not None, inc=inc)
            v = speeds(tr)
            code, collide = touch_codes(tr, t, thr["force"][str(t)]["environment"])
            if bool(collide.any()) != (r["max_contact_force"]["environment"] > thr["force"][str(t)]["environment"]):
                raise RuntimeError(f"{cond} {k}: per-step collision flags disagree with the record's peak force")
            rows.append({"condition": cond, "task": t, "init": k[1], "success": bool(r["success"]), "steps": r["steps"],
                         "outcome_v2": c["outcome"], "subtype_v2": c["subtype"],
                         "outcome_dest": dst.label(c["outcome"], t, head),
                         "heading": None if head is None else head["heading"],
                         "approach": None if head is None else {str(a): round(b, 4) for a, b in head["approach"].items()},
                         "collision": bool(r["max_contact_force"]["environment"] > thr["force"][str(t)]["environment"]),
                         "off_table": bool(r["off_table"]),
                         "mean_speed": float(v.mean()) if len(v) else 0.0})
            traces[(cond, k)] = {"eef": tr["eef_pos"], "speed": v, "touch": code, "collide": collide,
                                 "outcome": rows[-1]["outcome_dest"]}
    return rows, traces, skipped, dest
