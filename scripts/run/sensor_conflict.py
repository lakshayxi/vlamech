"""Sensor conflict, open loop: the policy queried on mixes of the real world and a counterfactual one (protocol sensor_conflict).

Usage:  python scripts/run/sensor_conflict.py --split exploratory|confirmatory [--shard i/n] [--limit N] [--no-patching]
  -> results/sensor_conflict/<split>/<task>-<init>.npz (one file per episode; resumable, atomic writes)
     + manifest.shard<i>of<n>.json

For every successful clean episode of sensor_conflict.tasks and each of its four phases (sensor_conflict.phases), the
captured simulator state is restored and rendered as it is (the real world) and with one physical fact changed
(vlamech.perturbations.worlds). The policy is then queried on every mix of the two worlds' third-person frame, wrist
frame and proprio (vlamech.analysis.arbitration), under the own instruction:
  object     the object to grasp (approach, pregrasp) or the destination (transport, place) moved; sources agentview, wrist
  distractor a non-task object moved the same distance; both cameras from the edited world only
  arm        the end effector moved; sources agentview, wrist, proprio; also with both cameras blurred / noised
  gripper    the gripper rendered open <-> closed; sources agentview, wrist, proprio
  stale      the captured observation of the previous query; sources agentview, wrist, proprio
  render     the real world's action from the re-render and from the captured observation (render noise floor)
For the object, arm and gripper edits the readout positions' mean state at the probe layers is stored for every input.
Patching (pregrasp and transport): single-point patches of the conflicting source's, the instruction's and the readout
positions with the real run's states, at the points of sensor_conflict.patch_grid.
Every forward is batch 1 (exact; batching changes bf16 results).

--patch-stale (held-out episodes already in results/sensor_conflict/confirmatory/): at pregrasp and transport, for each
single-source delay conflict (only that source from the previous query, 0.4 s old), single-point patches of that
source's, the instruction's and the readout positions with the current observation's states
  -> results/sensor_conflict/patching_stale/<task>-<init>.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.analysis.arbitration import subsets  # noqa: E402
from vlamech.config import protocol  # noqa: E402
from vlamech.envs.replay import load_capture, restore  # noqa: E402
from vlamech.interventions.patching import first_closing  # noqa: E402
from vlamech.perturbations import worlds as W  # noqa: E402
from vlamech.runs import identity  # noqa: E402
from vlamech.utils.io import atomic_write_json, atomic_write_npz, read_json  # noqa: E402
from vlamech.utils.seeding import derive_seed  # noqa: E402

OUT = ROOT / "results" / "sensor_conflict"
PHASES = ("approach", "pregrasp", "transport", "place")
PROBE_LAYERS = (8, 16, 24)
OBJECT_SHIFT_M, ARM_SHIFT_M = 0.05, 0.03
GRIPPER_OPEN_ABOVE_M = 0.04  # finger opening (m) above which the gripper counts as open; fully open is 0.08
PATCH_PHASES = ("pregrasp", "transport")
SOURCES3 = ("agentview", "wrist", "proprio")


def phase_queries(actions_norm: np.ndarray) -> dict | None:
    """{phase: captured query index} by protocol sensor_conflict.phases, or None if the episode does not qualify."""
    g = first_closing(actions_norm)
    if g is None or g < 2:
        return None
    grip = actions_norm[:, :, 6].mean(-1)
    later = [k for k in range(g + 1, len(grip)) if grip[k] >= 0.5]
    r = later[0] if later else len(grip) - 1
    if r - g < 3:
        return None
    return {"approach": g - 2, "pregrasp": g - 1, "transport": g + (r - g) // 2, "place": r - 1}


class Queries:
    """The policy on observation mixes; every input built from (agentview, wrist, proprio) arrays."""

    def __init__(self, policy, instruction: str):
        self.policy, self.ins = policy, instruction

    def inp(self, a, w, p):
        return self.policy.prepare(a, w, p, self.ins)

    def act(self, a, w, p, feats: bool = False):
        inp = self.inp(a, w, p)
        out = self.policy.forward(inp, output_hidden_states=feats)
        if not feats:
            return out["actions_norm"], None
        ro = np.array(inp.layout.readout)
        f = np.stack([out["hidden_states"][l + 1][0, ro].float().mean(0).cpu().numpy() for l in PROBE_LAYERS])
        return out["actions_norm"], f.astype(np.float16)

    def mixes(self, real: tuple, edited: tuple, sources: tuple, feats: bool = False):
        """Actions (2**n, 8, 7) [and probe features] for every mix; sources index into (agentview, wrist, proprio)."""
        acts, fs = [], []
        for bits in subsets(len(sources)):
            obs = list(real)
            for b, s in zip(bits, sources):
                if b:
                    obs[SOURCES3.index(s)] = edited[SOURCES3.index(s)]
            a, f = self.act(*obs, feats=feats)
            acts.append(a)
            fs.append(f)
        return np.stack(acts).astype(np.float32), (np.stack(fs) if feats else None)


def obs_of(frame) -> tuple:
    return (np.array(frame.agentview), np.array(frame.wrist), np.array(frame.proprio, np.float64))


def changed_pixels(policy, a: tuple, b: tuple) -> int:
    """Pixels (both 224 policy views) whose value differs by more than 8 levels in some channel."""
    va, vb = policy.preprocess_images(a[0], a[1]), policy.preprocess_images(b[0], b[1])
    return int(sum(np.any(np.abs(np.asarray(x, np.int16) - np.asarray(y, np.int16)) > 8, -1).sum() for x, y in zip(va, vb)))


def patch_curves(policy, q: Queries, real: tuple, mixed: tuple, source_group: str, grid: list) -> np.ndarray:
    """(3 groups [source, instruction, readout], len(grid), 8, 7): single-point patches of the real run's states."""
    from vlamech.interventions import patching as P

    caps, _ = P.donor(policy, q.inp(*real))
    inp = q.inp(*mixed)
    vf = policy.vision_features(inp)
    out = np.zeros((3, len(grid), 8, 7), np.float32)
    for gi, group in enumerate((source_group, "instruction", "readout")):
        pos = P.group_positions(inp.layout, group)
        for pi, point in enumerate(grid):
            out[gi, pi] = P.run(policy, inp, P.patches(caps, pos, [point]), vision_feats=vf)
    del caps
    return out


def episode(env, policy, cap: dict, acts: np.ndarray, key: str, instruction: str, targets: dict, corruptors: dict,
            do_patch: bool, grid: list) -> tuple[dict, dict]:
    phases = phase_queries(acts)
    if phases is None:
        return {}, {"skipped": "phases"}
    depth = float(protocol("sensor_conflict.max_penetration_m"))
    q = Queries(policy, instruction)
    arrays, meta = {}, {"phases": phases, "worlds": {}}
    manip = targets["manipulated_objects"][0]
    dest = [e for e in targets["own_entities"] if e != manip][0]
    for ph, k in phases.items():
        restore(env, cap, k, strict=False)
        base = W.World(env.get_sim_state())
        real = obs_of(W.render(env, base))
        info = meta["worlds"][ph] = {"query": int(k), "step": int(cap["step"][k])}
        # render noise floor: the re-rendered real world against the captured observation of the same query
        cap_obs = (cap["agentview"][k], cap["wrist"][k], cap["proprio"][k])
        arrays[f"{ph}__render__actions"] = np.stack([q.act(*real)[0], q.act(*cap_obs)[0]]).astype(np.float32)

        def seeded(edit):
            return W.directions(8, derive_seed("sensor_conflict", edit, key, ph))

        # object: the thing to grasp before the grasp, the destination after it
        entity = manip if ph in ("approach", "pregrasp") else dest
        wo = W.first_valid(env, base, lambda d: W.shift_entity(env, base, entity, OBJECT_SHIFT_M * d), seeded("object"), depth)
        if wo is not None:
            edited = obs_of(W.render(env, wo))
            arrays[f"{ph}__object__actions"], arrays[f"{ph}__object__feats"] = q.mixes(real, edited, ("agentview", "wrist"), feats=True)
            info["object"] = wo.detail | {"changed_px": changed_pixels(policy, real, edited)}
            if do_patch and ph in PATCH_PHASES:
                for m, cam in enumerate(("agentview", "wrist")):
                    mixed = list(real)
                    mixed[m] = edited[m]
                    arrays[f"{ph}__patch__{cam}"] = patch_curves(policy, q, real, tuple(mixed), cam, grid)
        # distractor: a non-task object moved the same distance, both cameras
        others = list(targets["non_target_objects"])
        order = np.random.default_rng(derive_seed("sensor_conflict", "distractor", key, ph)).permutation(len(others))
        for i in order:
            name = others[i]
            wd = W.first_valid(env, base, lambda d: W.shift_entity(env, base, name, OBJECT_SHIFT_M * d), seeded("distractor"), depth)
            if wd is not None:
                edited_d = obs_of(W.render(env, wd))
                arrays[f"{ph}__distractor__actions"] = q.mixes(real, edited_d, ("agentview", "wrist"))[0][[0, -1]]
                info["distractor"] = wd.detail | {"changed_px": changed_pixels(policy, real, edited_d)}
                break
        # arm: the end effector somewhere else (the held object moves with it after the grasp)
        carry = manip if ph in ("transport", "place") else None
        wa = W.first_valid(env, base, lambda d: W.shift_arm(env, base, ARM_SHIFT_M * d, carry=carry), seeded("arm"), depth)
        if wa is not None:
            edited_a = obs_of(W.render(env, wa))
            arrays[f"{ph}__arm__actions"], arrays[f"{ph}__arm__feats"] = q.mixes(real, edited_a, SOURCES3, feats=True)
            info["arm"] = wa.detail
            for label, corr in corruptors.items():
                step = int(cap["step"][k])
                cr = tuple(corr(c, x, step) for c, x in zip(("agentview", "wrist"), real[:2])) + (real[2],)
                ce = tuple(corr(c, x, step) for c, x in zip(("agentview", "wrist"), edited_a[:2])) + (edited_a[2],)
                arrays[f"{ph}__arm-{label}__actions"] = q.mixes(cr, ce, SOURCES3)[0]
            if do_patch and ph in PATCH_PHASES:
                arrays[f"{ph}__patch__proprio"] = patch_curves(policy, q, real, (real[0], real[1], edited_a[2]), "proprio", grid)
        # gripper: rendered open <-> closed
        wg = W.flip_gripper(env, base, GRIPPER_OPEN_ABOVE_M)
        if not (W.penetrations(env, wg, depth) - W.penetrations(env, base, depth)):
            edited_g = obs_of(W.render(env, wg))
            arrays[f"{ph}__gripper__actions"], arrays[f"{ph}__gripper__feats"] = q.mixes(real, edited_g, SOURCES3, feats=True)
            info["gripper"] = wg.detail
        # stale: the previous query's captured observation against this one's
        if k >= 1:
            now = (cap["agentview"][k], cap["wrist"][k], cap["proprio"][k])
            prev = (cap["agentview"][k - 1], cap["wrist"][k - 1], cap["proprio"][k - 1])
            arrays[f"{ph}__stale__actions"] = q.mixes(now, prev, SOURCES3)[0]
    return arrays, meta


def patch_stale(policy, src, out: Path, keys: list, grid: list, suite):
    out.mkdir(parents=True, exist_ok=True)
    for n, key in enumerate(keys):
        path = out / f"{key}.npz"
        if path.exists():
            continue
        t = int(key[4:].split("-")[0])
        cap = load_capture(src / "captures" / f"{key}.npz")
        phases = phase_queries(np.load(src / "actions" / f"{key}.npz")["actions_norm"])
        q = Queries(policy, suite.get_task(t).language)
        arrays = {}
        for ph in PATCH_PHASES:
            k = phases[ph]
            now = (cap["agentview"][k], cap["wrist"][k], cap["proprio"][k])
            prev = (cap["agentview"][k - 1], cap["wrist"][k - 1], cap["proprio"][k - 1])
            arrays[f"{ph}__now"] = q.act(*now)[0]
            for m, source in enumerate(SOURCES3):
                mixed = list(now)
                mixed[m] = prev[m]
                arrays[f"{ph}__mixed__{source}"] = q.act(*mixed)[0]
                arrays[f"{ph}__patch__{source}"] = patch_curves(policy, q, now, tuple(mixed), source, grid)
        atomic_write_npz(path, **arrays)
        print(f"[patch_stale] {key} ({n + 1}/{len(keys)})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("exploratory", "confirmatory"), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes only (development)")
    ap.add_argument("--no-patching", action="store_true")
    ap.add_argument("--patch-stale", action="store_true", help="patching of the delay conflicts only (see above)")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    src = ROOT / protocol("sensor_conflict.captures")[a.split]
    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    mf = out / f"manifest.shard{si}of{sn}.json"
    meta = {"source": str(src.relative_to(ROOT)), "patching": not a.no_patching,
            "started": datetime.now().isoformat(timespec="seconds"), "code": identity.code_identity(strict=False)["project"]}
    atomic_write_json(mf, [*(read_json(mf) if mf.exists() else []), meta])

    from libero.libero import benchmark

    from vlamech.envs.libero_env import LiberoTaskEnv
    from vlamech.models.oft import OFTPolicy
    from vlamech.perturbations.image import CorruptionSpec, EpisodeCorruptor
    from vlamech.utils.gpu import require_gpu

    suite = benchmark.get_benchmark_dict()[protocol("env.suite")]()
    tasks = set(protocol("sensor_conflict.tasks"))
    ok = {}
    for f in sorted(src.glob("episodes.shard*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            ok[f"task{r['task_id']}-init{r['init_state']}"] = r["success"] and r["error"] is None
    keys = [k for k in sorted(ok, key=lambda k: (int(k[4:].split("-")[0]), int(k.split("init")[1])))
            if ok[k] and int(k[4:].split("-")[0]) in tasks]
    keys = keys[: a.limit] if a.limit else keys
    keys = [k for i, k in enumerate(keys) if i % sn == si]
    grid = list(protocol("sensor_conflict.patch_grid"))
    vision = [c for c in protocol("sensor_conflict.vision_conditions") if c != "identity"]
    require_gpu(20)
    repo = protocol("policy.checkpoint")
    model = identity.verify_model_files(repo)
    policy = OFTPolicy(repo, revision=model["revision"], center_crop=protocol("policy.center_crop"))
    if a.patch_stale:
        done = {f.stem for f in (OUT / "confirmatory").glob("task*-init*.npz")}
        patch_stale(policy, src, OUT / "patching_stale", [k for k in keys if k in done], grid, suite)
        return
    envs = {}
    print(f"[sensor_conflict] {a.split}: {len(keys)} episodes", flush=True)
    for n, key in enumerate(keys):
        path = out / f"{key}.npz"
        if path.exists():
            continue
        t = int(key[4:].split("-")[0])
        if t not in envs:
            envs[t] = LiberoTaskEnv(protocol("env.suite"), t, resolution=int(protocol("env.resolution")))
        env = envs[t]
        cap = load_capture(src / "captures" / f"{key}.npz")
        acts = np.load(src / "actions" / f"{key}.npz")["actions_norm"]
        corruptors = {}
        for label in vision:
            fam, val = label.split("-")
            corruptors[label] = EpisodeCorruptor(CorruptionSpec(fam, float(val)), key, None)
        t0 = datetime.now()
        arrays, emeta = episode(env, policy, cap, acts, key, suite.get_task(t).language, env.task_targets(t),
                                corruptors, not a.no_patching, grid)
        emeta |= {"key": key, "task": t, "instruction": suite.get_task(t).language}
        atomic_write_npz(path, meta=np.array(json.dumps(emeta)), **arrays)
        print(f"[sensor_conflict] {key} ({n + 1}/{len(keys)}) {len(arrays)} arrays "
              f"{(datetime.now() - t0).total_seconds():.0f}s {emeta.get('skipped', '')}", flush=True)


if __name__ == "__main__":
    main()
