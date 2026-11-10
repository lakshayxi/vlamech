"""Closed-loop LIBERO episodes under one visual condition and one instruction mode.

Usage (after `source env/env.sh`):
  python scripts/run/closed_loop.py --experiment baseline --states all --replicate 0
  python scripts/run/closed_loop.py --experiment instruction_reliance --states all --instruction other_task
  python scripts/run/closed_loop.py --experiment severity_calibration --states calibration --family noise --value 16
  python scripts/run/closed_loop.py --experiment corruption_sweep --states confirmatory --family blur --value 8
  python scripts/run/closed_loop.py --experiment cross_model --states confirmatory --policy pi05   # LeRobot env
  python scripts/run/closed_loop.py --experiment debug ...   # anything, from uncommitted code, into results/debug/
Parallel shards: --shard 0/3, --shard 1/3, --shard 2/3 (optionally sharing one model, scripts/run/served.sh).

Before a model is loaded, the request is validated against its experiment (vlamech.runs.experiments), runtime
settings are resolved from the protocol, and code, upstream checkouts, model files and assets are verified and
written into an immutable manifest. A run resumes only with an identical manifest; each shard is owned by one process.

Outputs in results/<experiment>/<run>/: manifest.json, sessions.jsonl, topology.json, episodes*.jsonl (each record
stamped with the manifest digest), traces/<key>.npz (per-step oracle trace, always), and, as the experiment
requires, videos/<key>.mp4, actions/<key>.npz and captures/<key>.npz (every query's exact inputs). Episodes whose
code raised are retried once, then written with `error` set; analyses exclude and count them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.config import protocol, split_states  # noqa: E402
from vlamech.envs.trace import TRACE_SCHEMA  # noqa: E402
from vlamech.perturbations.image import NEEDS_BACKGROUND, NEEDS_TARGET, CorruptionSpec  # noqa: E402
from vlamech.runs import experiments, identity, manifest as mf  # noqa: E402
from vlamech.utils.io import PROJECT_ROOT, append_jsonl, atomic_write_npz, file_sha256, repair_jsonl  # noqa: E402
from vlamech.utils.seeding import set_global_seed  # noqa: E402

INSTRUCTION_MODES = ("own", "other_task", "empty")
GPU_GIB_PER_PROCESS = 20  # 7.5B bf16 weights (~15 GiB) plus activations


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(experiments.EXPERIMENTS))
    ap.add_argument("--policy", default="oft", choices=experiments.POLICIES)
    ap.add_argument("--states", required=True, help="calibration | exploratory | confirmatory | all | a-b")
    ap.add_argument("--tasks", default="0-9", help="a-b")
    ap.add_argument("--family", default="identity")
    ap.add_argument("--value", type=float, default=None)
    ap.add_argument("--cameras", default="agentview,wrist")
    ap.add_argument("--exact-model-input", action="store_true",
                    help="background-only families: composite after OFT preprocessing")
    ap.add_argument("--instruction", default="own", choices=INSTRUCTION_MODES)
    ap.add_argument("--replicate", type=int, default=0, help="independent repeat of the same condition")
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs episodes with index %% n == i")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--mitigation", default=None, help="cag:<omega>[:stop<threshold>] | horizon:<k>[:ensemble]")
    ap.add_argument("--record-actions", action="store_true")
    ap.add_argument("--record-video", action="store_true", help="videos/<key>.mp4: scene and exact network inputs")
    ap.add_argument("--capture-queries", action="store_true",
                    help="captures/<key>.npz: every query's frames, proprio and simulator state")
    ap.add_argument("--policy-server", type=Path, default=None, metavar="SOCKET",
                    help="query a running scripts/run/serve_policy.py instead of loading the model in this process")
    ap.add_argument("--accept-code-change", default=None, metavar="WHY",
                    help="resume although the code commit changed; the justification is recorded")
    return ap.parse_args(argv)


def _range(text: str, n: int) -> list[int]:
    if text in ("calibration", "exploratory", "confirmatory"):
        return split_states(text)
    if text == "all":
        return list(range(n))
    lo, hi = (int(x) for x in text.split("-"))
    if not 0 <= lo <= hi < n:
        raise ValueError(f"range {text} outside 0-{n - 1}")
    return list(range(lo, hi + 1))


def instruction_for(mode: str, task_id: int, suite) -> str:
    if mode == "own":
        return suite.get_task(task_id).language
    if mode == "other_task":  # protocol instructions.wrong_instruction_rule
        return suite.get_task((task_id + 5) % suite.n_tasks).language
    if mode == "empty":
        return ""
    raise ValueError(mode)


def rollout_video(path, instruction, condition, suite, task_id, settings, view_fn=None):
    """(RolloutVideo, frame_hook) for one episode; the final frame states the task's outcome."""
    from vlamech.viz.video import RolloutVideo

    names = {t: suite.get_task(t).language for t in range(suite.n_tasks)}
    inputs = ("agentview", "wrist")[: int(settings.get("policy.num_images", 2))]
    video = RolloutVideo(path, instruction, condition, names, int(settings["env.max_steps"]),
                         settings.get("policy.center_crop", True), inputs, view_fn=view_fn)

    def hook(step, frame, fed, fresh, goals):
        final = None
        if fed is None:  # after the last step
            final = (f"SUCCESS: goal of task {task_id} reached after {step} steps" if goals[task_id]
                     else f"NOT ACHIEVED: task {task_id} goal false after {step} steps")
        video.add(step, frame, fed, fresh, goals, final)

    return video, hook


def load_policy(args, settings, model):
    from vlamech.utils.gpu import require_gpu

    repo = settings["policy.checkpoint"]
    if args.policy == "pi05":
        from vlamech.models.pi05 import Pi05Policy

        require_gpu(GPU_GIB_PER_PROCESS)
        policy = Pi05Policy(repo, revision=model["revision"], seed_namespace=f"rep{args.replicate}",
                            n_action_steps=int(settings["policy.n_action_steps"]))
        policy.check_settings(settings)
        return policy, policy.view
    if args.policy_server is not None:
        from vlamech.runs.policy_server import RemotePolicy

        return RemotePolicy(args.policy_server, {"repo": repo, "revision": model["revision"],
                                                 "center_crop": bool(settings["policy.center_crop"]),
                                                 "global_seed": int(settings["policy.global_seed"])}), None
    from vlamech.models.oft import OFTPolicy

    require_gpu(GPU_GIB_PER_PROCESS)
    return OFTPolicy(repo, revision=model["revision"], center_crop=settings["policy.center_crop"]), None


def main(argv=None):
    args = parse_args(argv)
    from libero.libero import benchmark

    from vlamech.interventions import mitigation as mit

    ex = experiments.EXPERIMENTS[args.experiment]
    # evidence the experiment requires is always recorded, so a forgotten flag cannot lose data
    args.record_video |= "video" in ex.records
    args.record_actions |= "actions" in ex.records
    args.capture_queries |= "captures" in ex.records
    settings = experiments.resolve_settings(args.experiment, args.policy)
    suite = benchmark.get_benchmark_dict()[settings["env.suite"]]()
    n_states = int(protocol("splits.n_init_states"))
    tasks = _range(args.tasks, suite.n_tasks)
    states = _range(args.states, n_states)
    spec = CorruptionSpec(args.family, args.value, tuple(c for c in args.cameras.split(",") if c),
                          exact_model_input=args.exact_model_input)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if not (shard_n >= 1 and 0 <= shard_i < shard_n):
        raise ValueError(f"invalid --shard {args.shard}")
    action_fn, mitigation = mit.parse(args.mitigation)
    if args.policy != "oft" and (mitigation or spec.exact_model_input or args.policy_server is not None):
        raise ValueError("mitigations, the exact-model-input variant and the policy server are OFT only")
    pre = experiments.preflight(args.experiment, states, spec, args.instruction, n_states, mitigation, args.policy)

    # identity: refuses on any mismatch; debug may run from a dirty tree
    code = identity.code_identity(strict=not ex.allow_dirty, policy=args.policy)
    if code["project"]["dirty"] and not ex.allow_dirty:
        raise RuntimeError("runs other than debug require a clean, committed working tree")
    model = identity.verify_model_files(settings["policy.checkpoint"])
    assets = {"fixture_layouts": identity.files_sha256(
        [PROJECT_ROOT / "assets" / f"fixture_layouts_{settings['env.suite']}.npz"])}
    # masks are computed when the family needs them and whenever queries are captured (clean runs included), so any
    # corruption can be re-created offline on captured states
    needs_masks = spec.family in NEEDS_BACKGROUND | NEEDS_TARGET or args.capture_queries
    request = {"experiment": args.experiment, "suite": settings["env.suite"], "tasks": tasks, "states": states,
               "condition": spec.to_dict(), "instruction_mode": args.instruction, "replicate": args.replicate,
               "record_actions": args.record_actions, "record_video": args.record_video,
               "capture_queries": args.capture_queries, "trace_schema": TRACE_SCHEMA, "mitigation": mitigation,
               "policy": args.policy, "policy_server": args.policy_server is not None}

    from vlamech.envs.episode import run_episode
    from vlamech.envs.libero_env import LiberoTaskEnv

    envs: dict = {}

    def env_cache(task_id, segmentation=needs_masks):
        if (task_id, segmentation) not in envs:
            envs[(task_id, segmentation)] = LiberoTaskEnv(
                settings["env.suite"], task_id, resolution=settings["env.resolution"], segmentation=segmentation,
                settle_steps=settings["env.settle_steps"], env_seed=settings["env.env_seed"])
        return envs[(task_id, segmentation)]

    sides = None
    if spec.family == "bg_patches":
        from vlamech.perturbations.assets import BG_SIDES_PATH, bg_side_lengths

        sides = bg_side_lengths(lambda t: env_cache(t, segmentation=True))
        assets["bg_patch_sides"] = identity.files_sha256([BG_SIDES_PATH])
    man = mf.build_manifest(request, settings, pre["frozen"], assets, code, model)

    out_root = ROOT / "results" / ("debug" if args.experiment == "debug" else args.experiment)
    mit_tag = "" if not mitigation else "__" + mit.label(mitigation).replace(":", "-") + (
        f"-stop{mitigation['safe_stop_below']:g}" if mitigation.get("safe_stop_below") is not None else "")
    chunk = mitigation["steps"] if mitigation and mitigation.get("method") == "horizon" else settings["env.open_loop_steps"]
    pol_tag = "" if args.policy == "oft" else f"__{args.policy}"
    run_name = args.run_name or f"{spec.tag}__ins-{args.instruction}{mit_tag}{pol_tag}__rep{args.replicate}"
    handle = mf.open_run(out_root / run_name, man, shard_i, shard_n, args.accept_code_change)
    try:
        dropped = repair_jsonl(handle.log)  # only this process owns this shard's file
        if dropped:
            print(f"[run] repaired {handle.log.name}: dropped {dropped} truncated record", flush=True)
        done = handle.done_keys()
        todo = [(t, s) for t in tasks for s in states]
        todo = [ep for k, ep in enumerate(todo) if k % shard_n == shard_i and ep not in done]
        print(f"[run] {handle.run_dir.relative_to(ROOT)}  shard {shard_i}/{shard_n}: {len(todo)} to run, "
              f"{len(done)} done; manifest {man['digest'][:12]}", flush=True)
        if not todo:
            return
        from vlamech.perturbations.image import EpisodeCorruptor

        set_global_seed(int(settings["policy.global_seed"]))
        policy, view_fn = load_policy(args, settings, model)
        t_start, n_done = time.time(), 0
        for task_id, init in todo:
            env = env_cache(task_id)
            ins = instruction_for(args.instruction, task_id, suite)
            key = f"task{task_id}-init{init}"
            for attempt in (1, 2):
                corruptor = EpisodeCorruptor(spec, key, sides)
                video, hook = None, None
                if args.record_video:
                    video, hook = rollout_video(handle.run_dir / "videos" / f"{key}.mp4", ins, spec.tag, suite,
                                                task_id, settings, view_fn)
                res, extra = run_episode(env, policy, init, corruptor, instruction=ins,
                                         mask_fn=env.masks if needs_masks else None, record_actions=args.record_actions,
                                         condition=spec.tag, max_steps=settings["env.max_steps"], chunk=chunk,
                                         frame_hook=hook, capture=args.capture_queries, action_fn=action_fn)
                if video is not None:
                    video.close(keep=res.error is None)
                if res.error is None:
                    break
                print(f"[run] {key} attempt {attempt} raised:\n{res.error}", flush=True)
            trace_sha = capture_sha = None
            if res.error is None:  # files are written before the record that references them
                tpath = handle.run_dir / "traces" / f"{key}.npz"
                atomic_write_npz(tpath, **extra["trace"])
                trace_sha = file_sha256(tpath)
                if args.capture_queries:
                    cpath = handle.run_dir / "captures" / f"{key}.npz"
                    atomic_write_npz(cpath, **extra["capture"])
                    capture_sha = file_sha256(cpath)
                if args.record_actions:
                    atomic_write_npz(handle.run_dir / "actions" / f"{key}.npz",
                                     actions_norm=np.stack(extra["actions_norm"]), query_steps=np.array(extra["query_steps"]))
            rec = handle.stamp(res.to_dict() | {"instruction_mode": args.instruction, "replicate": args.replicate,
                                                "corruptor_flags": corruptor.flags, "attempts": attempt,
                                                "trace_sha256": trace_sha, "capture_sha256": capture_sha,
                                                "action_info": extra["action_info"]})
            append_jsonl(handle.log, rec)
            n_done += 1
            rate = (time.time() - t_start) / n_done
            print(f"[run] {key} success={res.success} steps={res.steps} {res.wall_s:.1f}s "
                  f"({n_done}/{len(todo)}, eta {rate * (len(todo) - n_done) / 60:.0f} min)", flush=True)
    finally:
        handle.close()
        for e in envs.values():
            e.close()


if __name__ == "__main__":
    main()
