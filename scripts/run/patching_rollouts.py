"""Activation patching at states the corrupted rollouts actually reached (protocol patching.rollout_*).

Usage:  python scripts/run/patching_rollouts.py [--shard i/n] [--limit N]
  -> results/patching_rollouts/<condition>/<task>-<init>-q<k>.npz  (resumable; atomic writes)

Sources: the held-out corruption-sweep rollouts of patching.patch_conditions (rep0), bowl tasks. Each saved query holds the frame fed to the
policy (fed_agentview, fed_wrist) and the clean render of the same simulator state (agentview, wrist), so both runs of a
patch come from the rollout itself. States: offsets 0 and +1 from the first chunk the rollout's own commanded gripper
closes (the rule of patching.bowl_states applied to the rollout). Under the 3 bowl instructions, at the last layer only:
  prefix         clean instruction states clamped into the corrupted run
  donor          another episode's clean instruction states clamped into the corrupted run
  noising        corrupted instruction states injected into the clean run
  noising_donor  another episode's clean instruction states injected into the clean run
The other episode is the next init state of the same task and condition with a state at the same offset. An identity
patch is checked bit-exact on every state; every forward is batch 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vlamech.analysis.sweep import conditions, label_of  # noqa: E402
from vlamech.config import protocol  # noqa: E402
from vlamech.interventions.patching import bowl_instructions, bowl_queries, random_donor_key  # noqa: E402
from vlamech.runs import identity  # noqa: E402
from vlamech.utils.io import atomic_write_npz, read_jsonl  # noqa: E402

OUT = ROOT / "results" / "patching_rollouts"


def select(src: Path, tasks: set) -> dict:
    states = {}
    for f in sorted((src / "captures").glob("task*-init*.npz")):
        key = f.stem
        if int(key.split("-")[0][4:]) not in tasks:
            continue
        states[key] = bowl_queries(np.load(src / "actions" / f"{key}.npz")["actions_norm"], len(np.load(f)["step"]))
    return states


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="stop after N states (smoke test)")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))

    import torch
    from libero.libero import benchmark

    from vlamech.interventions import patching as P
    from vlamech.models.oft import OFTPolicy
    from vlamech.utils.gpu import require_gpu

    suite = benchmark.get_benchmark_dict()[protocol("env.suite")]()
    specs = {label_of(s): s for s in conditions()}
    tasks = set(protocol("statistics.bowl_tasks"))
    require_gpu(20)
    repo = protocol("policy.checkpoint")
    policy = OFTPolicy(repo, revision=identity.verify_model_files(repo)["revision"], center_crop=protocol("policy.center_crop"))
    done = 0
    for c in protocol("patching.patch_conditions"):
        src = ROOT / "results" / "corruption_sweep" / f"{specs[c].tag}__ins-own__rep0"
        out = OUT / c
        out.mkdir(parents=True, exist_ok=True)
        success = {}
        for f in sorted(src.glob("episodes*.jsonl")):
            for r in read_jsonl(f):
                if r.get("error") is None:
                    success[f"task{r['task_id']}-init{r['init_state']}"] = bool(r["success"])
        states = select(src, tasks)
        keys = [k for i, k in enumerate(sorted(states)) if i % sn == si]
        print(f"[rollouts] {c}: {len(keys)} episodes, {sum(len(states[k]) for k in keys)} states; no grasp "
              f"{sum(not states[k] for k in keys)}", flush=True)
        for key in keys:
            todo = [k for k in states[key] if not (out / f"{key}-q{k}.npz").exists()]
            if not todo:
                continue
            cap = np.load(src / "captures" / f"{key}.npz", allow_pickle=False)
            ins = bowl_instructions(int(cap["task_id"]), suite, protocol("statistics.bowl_tasks"))
            for k in todo:
                off = states[key].index(k)
                dkey = random_donor_key(states, key, off)
                if dkey is None:
                    print(f"[rollouts] {c} {key} q{k}: no donor at offset {off}, skipped", flush=True)
                    continue
                dcap = np.load(src / "captures" / f"{dkey}.npz", allow_pickle=False)
                dk = states[dkey][off]
                prep = {}
                for name, av, wr, pr in (("clean", cap["agentview"][k], cap["wrist"][k], cap["proprio"][k]),
                                         ("corr", cap["fed_agentview"][k], cap["fed_wrist"][k], cap["proprio"][k]),
                                         ("donor", dcap["agentview"][dk], dcap["wrist"][dk], dcap["proprio"][dk])):
                    first = policy.prepare(av, wr, pr, ins[0])
                    vf = policy.vision_features(first)
                    prep[name] = ([first] + [policy.prepare(av, wr, pr, i, images_224=first.images_224) for i in ins[1:]], vf)
                caps, base = {n: [] for n in prep}, {n: [] for n in prep}
                for n, (inps, vf) in prep.items():
                    for inp in inps:
                        cp, o = P.donor(policy, inp, vf)
                        caps[n].append(cp)
                        base[n].append(o["actions_norm"])
                (ic, vc), (ir, vr) = prep["clean"], prep["corr"]
                last = P.points_up_to(31)
                ident = P.run(policy, ir[0], P.patches(caps["corr"][0], P.group_positions(ir[0].layout, "vision"), ["embed", 15]), vr)
                if not np.array_equal(ident, base["corr"][0]):
                    raise RuntimeError(f"{c} {key} q{k}: identity patch is not exact")
                res = {n: np.zeros((len(ins), 8, 7), np.float32) for n in ("prefix", "donor_clamp", "noising", "noising_donor")}
                for j in range(len(ins)):
                    pos_r, pos_c = P.group_positions(ir[j].layout, "instruction"), P.group_positions(ic[j].layout, "instruction")
                    res["prefix"][j] = P.run(policy, ir[j], P.patches(caps["clean"][j], pos_r, last), vr)
                    res["donor_clamp"][j] = P.run(policy, ir[j], P.patches(caps["donor"][j], pos_r, last), vr)
                    res["noising"][j] = P.run(policy, ic[j], P.patches(caps["corr"][j], pos_c, last), vc)
                    res["noising_donor"][j] = P.run(policy, ic[j], P.patches(caps["donor"][j], pos_c, last), vc)
                atomic_write_npz(out / f"{key}-q{k}.npz", condition=np.array(c), success=np.array(success[key]), offset=np.array(off),
                                 donor_state=np.array(f"{dkey}-q{dk}"), instructions=np.array(ins),
                                 clean=np.stack(base["clean"]), corr=np.stack(base["corr"]), **res)
                del caps
                torch.cuda.empty_cache()
                done += 1
                print(f"[rollouts] {c} {key} q{k}", flush=True)
                if a.limit and done >= a.limit:
                    print("[rollouts] limit reached", flush=True)
                    return
    print("[rollouts] done", flush=True)


if __name__ == "__main__":
    main()
