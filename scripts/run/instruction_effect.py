"""Open-loop pass: every captured clean state, every frozen condition re-created offline, actions under every instruction.

Usage:  python scripts/run/instruction_effect.py --source <run dir with clean captures> [--policy-server SOCKET] [--shard i/n]
  -> results/instruction_effect/<source run name>/<task>-<init>.npz  (+ manifest.shard<i>of<n>.json, one entry per start)

For each captured query k of a CLEAN run (the state the policy was in, its raw frames, proprio and masks), for each
condition of configs/frozen_levels.yaml: the corruption is re-created exactly as the closed loop would have applied it
at that state and step (same EpisodeCorruptor key and step, captured masks; exact-model-input composited after
preprocessing), then the policy gives its action chunk for the instructed task, the 9 other tasks
(protocol metrics.alternative_instructions = all_other_tasks) and the empty instruction (CAG's unconditional
branch), sharing one vision pass (OFTPolicy.act_many).

Stored per episode: actions_norm (queries, conditions, instructions, 8, 7) float32, plus the conditions,
instructions, query steps and the source capture's SHA-256. Metrics are computed from these files by
scripts/analysis/instruction_effect.py. Conditions whose masks are missing in the source capture are skipped and listed.
"""
from __future__ import annotations

import argparse
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
from vlamech.perturbations.image import NEEDS_BACKGROUND, NEEDS_TARGET, EpisodeCorruptor  # noqa: E402
from vlamech.runs import identity  # noqa: E402
from vlamech.utils.io import atomic_write_json, atomic_write_npz, file_sha256, read_json  # noqa: E402

OUT = ROOT / "results" / "instruction_effect"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="run directory of a clean run with captures/")
    ap.add_argument("--policy-server", type=Path, default=None)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    man = read_json(a.source / "manifest.json")
    if man["request"]["condition"]["family"] != "identity":
        raise RuntimeError("the source must be a clean (identity) run")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[man["request"]["suite"]]()
    specs = conditions()
    out_dir = OUT / a.source.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source": str(a.source), "source_manifest_digest": man["digest"],
            "conditions": [label_of(s) for s in specs], "code": identity.code_identity(strict=False)["project"],
            "started": datetime.now().isoformat(timespec="seconds")}
    mf = out_dir / f"manifest.shard{si}of{sn}.json"
    starts = read_json(mf) if mf.exists() else []  # one entry per (re)start: a resume never hides the earlier code
    atomic_write_json(mf, [*(starts if isinstance(starts, list) else [starts]), meta])

    repo = protocol("policy.checkpoint")
    model = identity.verify_model_files(repo)
    if a.policy_server is not None:
        from vlamech.runs.policy_server import RemotePolicy

        policy = RemotePolicy(a.policy_server, {"repo": repo, "revision": model["revision"],
                                                "center_crop": bool(protocol("policy.center_crop")),
                                                "global_seed": int(protocol("policy.global_seed"))})
    else:
        from vlamech.models.oft import OFTPolicy
        from vlamech.utils.gpu import require_gpu

        require_gpu(20)
        policy = OFTPolicy(repo, revision=model["revision"], center_crop=protocol("policy.center_crop"))
    sides = None
    if any(s.family == "bg_patches" for s in specs):
        from vlamech.perturbations.assets import bg_side_lengths

        def no_env(_):
            raise RuntimeError("bg patch sides are not cached: run a bg_patches condition first (runner builds them)")

        sides = bg_side_lengths(no_env)  # the cached, metadata-checked sides the closed loop used
    files = sorted((a.source / "captures").glob("task*-init*.npz"))
    todo = [f for i, f in enumerate(files) if i % sn == si and not (out_dir / f.name).exists()]
    print(f"[open-loop] {len(todo)} episodes to do ({len(files)} captured), {len(specs)} conditions", flush=True)
    for f in todo:
        cap = load_capture(f)
        key = f.stem
        t = int(cap["task_id"])
        ins = instructions_for(t, suite)
        usable = [s for s in specs if s.family not in NEEDS_BACKGROUND | NEEDS_TARGET
                  or "mask_agentview_background" in cap]
        acts = np.full((len(cap["step"]), len(specs), len(ins), 8, 7), np.nan, np.float32)
        for j, spec in enumerate(specs):
            if spec not in usable:
                continue
            corr = EpisodeCorruptor(spec, key, sides)  # one per (episode, condition), queries in order
            for k in range(len(cap["step"])):
                av, wr, v224 = recreate(corr, cap, k, policy.center_crop)
                if v224 is None:
                    out = policy.act_many(av, wr, cap["proprio"][k], ins)
                else:  # exact-model-input: the preprocessed views are fed directly, one instruction at a time
                    out = {i: policy.forward(policy.prepare(av, wr, cap["proprio"][k], i, images_224=v224)) for i in ins}
                for n, i in enumerate(ins):
                    acts[k, j, n] = out[i]["actions_norm"]
        atomic_write_npz(out_dir / f.name, actions_norm=acts, steps=np.asarray(cap["step"]),
                         conditions=np.array([label_of(s) for s in specs]),
                         usable=np.array([s in usable for s in specs]), instructions=np.array(ins),
                         source_sha256=file_sha256(f))
        print(f"[open-loop] {key}: {len(cap['step'])} queries, {sum(s in usable for s in specs)}/{len(specs)} conditions",
              flush=True)


if __name__ == "__main__":
    main()
