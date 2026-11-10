"""Activation patching: offline forwards with hooks on captured clean states (protocol patching.*).

Usage:  python scripts/run/activation_patching.py --split exploratory|confirmatory
                                 [--experiments camera,entity,noising_control,patch] [--shard i/n] [--data-root DIR]
  -> results/activation_patching/<split>/<experiment>/<task>-<init>-q<k>.npz  (+ manifest.shard<i>of<n>.json, one
     entry per start). Finished files are skipped (resumable); writes are atomic.

Sources: the clean captures of the split (exploratory: results/clean_capture/<run>; confirmatory: the clean repeat of
the corruption sweep, rep1) with their executed clean actions (`actions/`) for the grasp rule. States: protocol
patching.bowl_states. Conditions: protocol patching.conditions, each re-created at the captured state exactly as the
closed loop applied it (vlamech.envs.offline.recreate).

Experiments (each state x condition):
  camera  actions under own, the 9 others and empty for: clean, the full condition, wrist-only, agentview-only
  entity  actions under the 11 instructions with only the bowl or only the destination occluded; the captured target
          mask (bowl and destination together) is split between them by the segmentation of the restored simulator
          state (nearest labelled pixel). Target occlusion fills the bounding box of its mask, so each part hides its
          own box and together they do not cover the box of the whole mask (box areas are saved)
  noising_control  the clean instruction states of the random donor injected into the clean run, prefix clamps at
          the control points: the damage a mismatched but uncorrupted instruction state does on its own
  patch   (patching.patch_conditions) denoising single-layer and prefix-clamp patches by group and grid point,
          controls (random donor, norm-matched noise) at the control points, and noising of the instruction group,
          under the 3 bowl instructions
Every forward is batch 1 (exact; batching changes bf16 results). An identity patch is checked bit-exact per state.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.sweep import conditions, label_of  # noqa: E402
from vlamech.config import protocol  # noqa: E402
from vlamech.envs.offline import instructions_for, recreate  # noqa: E402
from vlamech.envs.replay import load_capture  # noqa: E402
from vlamech.interventions.patching import bowl_instructions, bowl_queries, random_donor_key  # noqa: E402
from vlamech.perturbations.image import EpisodeCorruptor, target_box  # noqa: E402
from vlamech.runs import identity  # noqa: E402
from vlamech.utils.io import atomic_write_json, atomic_write_npz, read_json  # noqa: E402
from vlamech.utils.seeding import derive_seed  # noqa: E402

SOURCES = {"exploratory": "results/clean_capture/identity-agentview+wrist__ins-own__rep0",
           "confirmatory": "results/corruption_sweep/identity-agentview+wrist__ins-own__rep1"}
OUT = ROOT / "results" / "activation_patching"
EXPERIMENTS = ("camera", "entity", "noising_control", "patch")  # cheap first
NOISE_SEED_KEY = "s5-noise"  # seed key of the published runs: changing it changes the norm-matched noise draws


def select_states(src: Path) -> dict:
    """{episode key: [bowl queries]} for the bowl-task episodes of a clean capture run (protocol patching.bowl_states)."""
    bowl_tasks = set(protocol("statistics.bowl_tasks"))
    out = {}
    for f in sorted((src / "captures").glob("task*-init*.npz")):
        if int(f.stem.split("-")[0][4:]) in bowl_tasks:
            out[f.stem] = bowl_queries(np.load(src / "actions" / f"{f.stem}.npz")["actions_norm"], len(np.load(f)["step"]))
    return out


def box_area(mask: np.ndarray, margin: float) -> int:
    b = target_box(mask, margin)
    return 0 if b is None else (b[1] - b[0]) * (b[3] - b[2])


def images(cap: dict, spec, key: str, queries: list, center_crop: bool) -> dict:
    """{k: (agentview, wrist)} under `spec` at the captured queries (corruptor called for every query in order)."""
    corr = EpisodeCorruptor(spec, key)
    out = {}
    for k in range(max(queries) + 1):
        av, wr, v224 = recreate(corr, cap, k, center_crop)
        if v224 is not None:
            raise RuntimeError("exact-model-input conditions are not in patching.conditions")
        if k in queries:
            out[k] = (av, wr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(SOURCES), required=True)
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--data-root", type=Path, default=ROOT)
    a = ap.parse_args()
    exps = [e for e in EXPERIMENTS if e in a.experiments.split(",")]
    si, sn = (int(x) for x in a.shard.split("/"))
    src = a.data_root / SOURCES[a.split]
    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    mf = out / f"manifest.shard{si}of{sn}.json"
    meta = {"source": str(src), "experiments": exps, "started": datetime.now().isoformat(timespec="seconds"),
            "code": identity.code_identity(strict=False)["project"]}
    atomic_write_json(mf, [*(read_json(mf) if mf.exists() else []), meta])

    import torch
    from libero.libero import benchmark

    from vlamech.interventions import patching as P
    from vlamech.models.oft import OFTPolicy
    from vlamech.utils.gpu import require_gpu

    suite = benchmark.get_benchmark_dict()[protocol("env.suite")]()
    specs = {label_of(s): s for s in conditions()}
    conds, patch_conds = list(protocol("patching.conditions")), list(protocol("patching.patch_conditions"))
    grid, groups = list(protocol("patching.layer_grid")), list(protocol("patching.groups"))
    prefix_groups, ctrl_points = list(protocol("patching.prefix_groups")), list(protocol("patching.control_points"))
    states = select_states(src)
    bowl_tasks = protocol("statistics.bowl_tasks")
    keys = [k for i, k in enumerate(sorted(states)) if i % sn == si]
    require_gpu(20)
    repo = protocol("policy.checkpoint")
    model = identity.verify_model_files(repo)
    policy = OFTPolicy(repo, revision=model["revision"], center_crop=protocol("policy.center_crop"))
    cc = policy.center_crop
    envs = {}
    print(f"[patching] {a.split}: {len(keys)} episodes, experiments {exps}, conditions {conds}", flush=True)

    for exp in exps:
        (out / exp).mkdir(exist_ok=True)
        for key in keys:
            qs = states[key]
            todo = [k for k in qs if not (out / exp / f"{key}-q{k}.npz").exists()]
            if not todo:
                continue
            cap = load_capture(src / "captures" / f"{key}.npz")
            t = int(cap["task_id"])
            clean = {k: (cap["agentview"][k], cap["wrist"][k]) for k in todo}
            need = {"patch": patch_conds, "camera": conds}.get(exp, [])
            imgs = {c: images(cap, specs[c], key, todo, cc) for c in need}

            if exp == "camera":
                ins = instructions_for(t, suite)
                for k in todo:
                    acts, labels = [], []
                    av, wr = clean[k]
                    o = policy.act_many(av, wr, cap["proprio"][k], ins)
                    acts.append([o[i]["actions_norm"] for i in ins])
                    labels.append("identity")
                    for c in conds:
                        variants = {"full": specs[c], "wrist": dataclasses.replace(specs[c], cameras=("wrist",)),
                                    "agentview": dataclasses.replace(specs[c], cameras=("agentview",))}
                        for v, spec in variants.items():
                            av, wr = images(cap, spec, key, [k], cc)[k] if v != "full" else imgs[c][k]
                            o = policy.act_many(av, wr, cap["proprio"][k], ins)
                            acts.append([o[i]["actions_norm"] for i in ins])
                            labels.append(f"{c}|{v}")
                    atomic_write_npz(out / exp / f"{key}-q{k}.npz", variants=np.array(labels), instructions=np.array(ins),
                                     bowl=np.array(True), actions=np.asarray(acts, np.float32))
                    print(f"[patching] camera {key} q{k}", flush=True)

            elif exp == "entity":
                from scipy import ndimage

                from vlamech.envs.libero_env import LiberoTaskEnv
                from vlamech.envs.replay import restore

                if t not in envs:
                    envs[t] = LiberoTaskEnv(protocol("env.suite"), t, resolution=int(protocol("env.resolution")),
                                            segmentation=True, settle_steps=int(protocol("env.settle_steps")),
                                            env_seed=int(protocol("env.env_seed")))
                env = envs[t]
                tt = env.task_targets(t)
                ids = env.instance_ids()
                manip = [ids[e] for e in tt["manipulated_objects"]]
                dest = [ids[e] for e in env.own_entities if e not in tt["manipulated_objects"]]
                ins = instructions_for(t, suite)
                spec = specs["target_occlusion"]
                for k in todo:
                    frame, report = restore(env, cap, k, strict=False)
                    parts, stats, boxes = {"bowl": {}, "destination": {}}, {}, {}
                    for cam in ("agentview", "wrist"):
                        seg = getattr(frame, f"{cam}_seg")
                        lab = np.full(seg.shape, -1, np.int8)
                        lab[np.isin(seg, manip)] = 0
                        lab[np.isin(seg, dest)] = 1
                        target = cap[f"mask_{cam}_target"][k]
                        if (lab >= 0).any():
                            _, (ri, ci) = ndimage.distance_transform_edt(lab < 0, return_indices=True)
                            near = lab[ri, ci]
                        else:
                            near = np.full(seg.shape, -1, np.int8)
                        parts["bowl"][cam] = target & (near == 0)
                        parts["destination"][cam] = target & (near == 1)
                        stats[cam] = [int(target.sum()), int(parts["bowl"][cam].sum()),
                                      int(parts["destination"][cam].sum())]
                        boxes[cam] = [box_area(m, float(protocol("corruption.target_occlusion_margin")))
                                      for m in (target, parts["bowl"][cam], parts["destination"][cam])]
                    acts = []
                    for part in ("bowl", "destination"):
                        c2 = dict(cap)
                        for cam in ("agentview", "wrist"):
                            m = cap[f"mask_{cam}_target"].copy()
                            m[k] = parts[part][cam]
                            c2[f"mask_{cam}_target"] = m
                        av, wr = images(c2, spec, key, [k], cc)[k]
                        o = policy.act_many(av, wr, cap["proprio"][k], ins)
                        acts.append([o[i]["actions_norm"] for i in ins])
                    atomic_write_npz(out / exp / f"{key}-q{k}.npz", parts=np.array(["bowl", "destination"]),
                                     instructions=np.array(ins), actions=np.asarray(acts, np.float32),
                                     pixels=np.array([stats["agentview"], stats["wrist"]]),  # [cam][target, bowl, dest]
                                     box_pixels=np.array([boxes["agentview"], boxes["wrist"]]),  # hidden box areas
                                     restore_px=np.array([report["agentview_px"], report["wrist_px"]]))
                    print(f"[patching] entity {key} q{k}: pixels {stats}", flush=True)

            elif exp == "noising_control":
                ins = bowl_instructions(t, suite, bowl_tasks)
                points = [P.points_up_to(p) for p in ctrl_points]
                for k in todo:
                    off = qs.index(k)
                    dkey = random_donor_key(states, key, off)
                    dcap = load_capture(src / "captures" / f"{dkey}.npz")
                    dk = states[dkey][off]
                    prep = {}
                    for name, (av, wr), pr in (("clean", clean[k], cap["proprio"][k]),
                                               ("donor", (dcap["agentview"][dk], dcap["wrist"][dk]), dcap["proprio"][dk])):
                        first = policy.prepare(av, wr, pr, ins[0])
                        prep[name] = ([first] + [policy.prepare(av, wr, pr, i, images_224=first.images_224) for i in ins[1:]],
                                      policy.vision_features(first))
                    donor_caps = [P.donor(policy, inp, prep["donor"][1])[0] for inp in prep["donor"][0]]
                    ic, vc = prep["clean"]
                    base, acts = [], np.zeros((len(points), len(ins), 8, 7), np.float32)
                    for j, inp in enumerate(ic):
                        own, o = P.donor(policy, inp, vc)
                        base.append(o["actions_norm"])
                        pos = P.group_positions(inp.layout, "instruction")
                        if j == 0 and not np.array_equal(P.run(policy, inp, P.patches(own, pos, points[-1]), vc), base[0]):
                            raise RuntimeError(f"{key} q{k}: identity patch is not exact")
                        for p_i, pts in enumerate(points):
                            acts[p_i, j] = P.run(policy, inp, P.patches(donor_caps[j], pos, pts), vc)
                        del own
                    del donor_caps
                    atomic_write_npz(out / exp / f"{key}-q{k}.npz", instructions=np.array(ins), clean=np.stack(base),
                                     control_points=np.array([str(p) for p in ctrl_points]), noising_control=acts,
                                     donor_state=np.array(f"{dkey}-q{dk}"))
                    print(f"[patching] noising_control {key} q{k} (donor {dkey}-q{dk})", flush=True)

            elif exp == "patch":
                ins = bowl_instructions(t, suite, bowl_tasks)
                for k in todo:
                    off = qs.index(k)
                    dkey = random_donor_key(states, key, off)
                    dcap = load_capture(src / "captures" / f"{dkey}.npz")
                    dk = states[dkey][off]
                    res = {}
                    for c in patch_conds:
                        prep = {}
                        for name, (av, wr), pr in (("clean", clean[k], cap["proprio"][k]), ("corr", imgs[c][k], cap["proprio"][k]),
                                                   ("donor", (dcap["agentview"][dk], dcap["wrist"][dk]), dcap["proprio"][dk])):
                            first = policy.prepare(av, wr, pr, ins[0])
                            vf = policy.vision_features(first)
                            inps = [first] + [policy.prepare(av, wr, pr, i, images_224=first.images_224) for i in ins[1:]]
                            prep[name] = (inps, vf)
                        caps = {n: [] for n in prep}
                        base = {n: [] for n in prep}
                        for n, (inps, vf) in prep.items():
                            for inp in inps:
                                cp, o = P.donor(policy, inp, vf)
                                caps[n].append(cp)
                                base[n].append(o["actions_norm"])
                        (ic, vc), (ir, vr) = prep["clean"], prep["corr"]
                        lay0 = ir[0].layout
                        ident = P.run(policy, ir[0], P.patches(caps["corr"][0], P.group_positions(lay0, "vision"),
                                                               ["embed", 15]), vr)
                        if not np.array_equal(ident, base["corr"][0]):
                            raise RuntimeError(f"{key} q{k} {c}: identity patch is not exact")
                        single = np.zeros((len(groups), len(grid), len(ins), 8, 7), np.float32)
                        prefix = np.zeros((len(prefix_groups), len(grid), len(ins), 8, 7), np.float32)
                        ctrl = np.zeros((2, len(prefix_groups), len(ctrl_points), len(ins), 8, 7), np.float32)
                        noising = np.zeros((2, len(grid), len(ins), 8, 7), np.float32)
                        for j, inp in enumerate(ir):
                            lay = inp.layout
                            for g_i, g in enumerate(groups):
                                pos = P.group_positions(lay, g)
                                for p_i, p in enumerate(grid):
                                    single[g_i, p_i, j] = P.run(policy, inp, P.patches(caps["clean"][j], pos, [p]), vr)
                            for g_i, g in enumerate(prefix_groups):
                                pos = P.group_positions(lay, g)
                                for p_i, p in enumerate(grid):
                                    pts = P.points_up_to(p)
                                    prefix[g_i, p_i, j] = P.run(policy, inp, P.patches(caps["clean"][j], pos, pts), vr)
                                for p_i, p in enumerate(ctrl_points):
                                    pts = P.points_up_to(p)
                                    ctrl[0, g_i, p_i, j] = P.run(policy, inp, P.patches(caps["donor"][j], pos, pts), vr)
                                    seed = derive_seed(NOISE_SEED_KEY, key, k, c, g, p, j)
                                    ctrl[1, g_i, p_i, j] = P.run(policy, inp, P.noise_patches(
                                        caps["clean"][j], caps["corr"][j], pos, pts, seed), vr)
                            pos = P.group_positions(ic[j].layout, "instruction")
                            for p_i, p in enumerate(grid):
                                noising[0, p_i, j] = P.run(policy, ic[j], P.patches(caps["corr"][j], pos, [p]), vc)
                                noising[1, p_i, j] = P.run(policy, ic[j], P.patches(caps["corr"][j], pos, P.points_up_to(p)), vc)
                        res[c] = {"clean": np.stack(base["clean"]), "corr": np.stack(base["corr"]),
                                  "donor": np.stack(base["donor"]), "single": single, "prefix": prefix, "ctrl": ctrl,
                                  "noising": noising}
                        del caps
                        torch.cuda.empty_cache()
                    atomic_write_npz(out / exp / f"{key}-q{k}.npz", conditions=np.array(patch_conds), instructions=np.array(ins),
                                     groups=np.array(groups), prefix_groups=np.array(prefix_groups),
                                     grid=np.array([str(p) for p in grid]), control_points=np.array([str(p) for p in ctrl_points]),
                                     donor_state=np.array(f"{dkey}-q{dk}"),
                                     **{f"{f}": np.stack([res[c][f] for c in patch_conds]) for f in res[patch_conds[0]]})
                    print(f"[patching] patch {key} q{k} (donor {dkey}-q{dk})", flush=True)
    print("[patching] done", flush=True)


if __name__ == "__main__":
    main()
