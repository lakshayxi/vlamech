"""One closed-loop episode under a visual condition, with the full outcome oracle.

Protocol (matches upstream OFT LIBERO eval): reset to benchmark init state -> 10 settle steps -> up to
max_steps policy steps, re-querying the policy every 8 steps (full chunk executed open-loop); the episode
ends early when the INSTRUCTED task's goal is satisfied (upstream `done`).

What is recorded per episode (all from simulator state, never from images):
  extra["trace"]         - the per-control-step oracle trace (vlamech.envs.trace; row 0 = post-settle frame).
                           Outcome classes (vlamech.metrics.outcome) are computed from it after collection,
                           with thresholds calibrated by rule; nothing threshold-dependent is decided here.
  summaries, all computed from the trace:
  success                - instructed goal satisfied
  goals_initial          - task ids whose goal already held at the post-settle frame (not events)
  goal_first_step        - {task_id: first step after which its goal predicate held} (other tasks too)
  obj_max_disp           - {object: max displacement from its post-settle position (m)}
  joint_max_disp         - {articulated joint: max |q - q_post-settle|} (drawers m, stove knob rad)
  max_contact_force      - {"object": max robot-object, "environment": max robot-table/fixture/wall}
                           contact normal force (N) (safety proxies; grasping is in "object")
  off_table              - objects that fell off the table (below the table-top plane AND outside its
                           footprint; LiberoTaskEnv.off_table) (safety proxy), in order of falling
  n_queries, steps, wall time, error (exceptions are recorded, not silently counted as failures)
Optionally the normalized action chunks per query (for open-loop analyses and monitors).

Measurement notes (limitations, stated so they are not mistaken for properties of the policy):
  * all oracle quantities are sampled once per control step (20 Hz), after the step; force peaks inside
    the physics substeps of one control step are not seen.
  * `success` is always the LOADED task's goal (upstream `done`), also when `instruction` overrides the
    prompt; whether the prompted task was achieved is read from goal_first_step.
  * an episode with `error` set is not a failure: analyses must exclude and report it (runner re-runs it).
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image

from vlamech.config import protocol
from vlamech.envs.libero_env import Frame, LiberoTaskEnv
from vlamech.envs.replay import QueryCapture
from vlamech.envs.trace import TRACE_SCHEMA, TraceRecorder
from vlamech.perturbations.image import EpisodeCorruptor


@dataclass
class EpisodeResult:
    task_id: int
    init_state: int
    condition: str
    instruction: str
    success: bool = False
    steps: int = 0
    n_queries: int = 0
    goals_initial: list = field(default_factory=list)
    goal_first_step: dict = field(default_factory=dict)
    obj_max_disp: dict = field(default_factory=dict)
    joint_max_disp: dict = field(default_factory=dict)
    max_contact_force: dict = field(default_factory=lambda: {"object": 0.0, "environment": 0.0})
    off_table: list = field(default_factory=list)
    wall_s: float = 0.0
    stopped_at: Optional[int] = None  # safe stop (mitigation): the step at which the robot handed back control
    trace_schema: int = TRACE_SCHEMA
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def run_episode(
    env: LiberoTaskEnv,
    policy,
    init_state: int,
    corruptor: EpisodeCorruptor,
    instruction: Optional[str] = None,
    mask_fn: Optional[Callable[[Frame], dict]] = None,
    action_fn: Optional[Callable] = None,
    record_actions: bool = False,
    condition: str = "",
    max_steps: Optional[int] = None,
    chunk: Optional[int] = None,
    frame_hook: Optional[Callable] = None,
    capture: bool = False,
) -> tuple[EpisodeResult, dict]:
    """Run one episode. `instruction` overrides the task's own (for wrong-instruction probes).

    mask_fn(frame) -> {"agentview": {"background": bool HxW, "target": bool HxW}, "wrist": {...}}; required only
    for families that need masks. action_fn(policy, frame_images, proprio, instruction) -> (actions (8,7)
    unnormalized, actions_norm, info dict); defaults to plain policy inference (mitigations plug in here, see
    vlamech.interventions.mitigation). info is recorded per query in extra["action_info"]; info["stop"] = True ends
    the episode before acting (safe stop; res.stopped_at = that step, never a success).
    frame_hook(step, frame, fed, fresh, goals) is called before each action is executed (frame = the state the
    action is applied in, fed = the images given to the policy at the latest query, fresh = queried at this step,
    goals = {task id: predicate} of that state) and once after the last step with fed=None; it only observes.
    capture=True stores every query's observation and simulator state in extra["capture"] (vlamech.envs.replay).
    """
    max_steps = int(protocol("env.max_steps") if max_steps is None else max_steps)
    chunk = int(protocol("env.open_loop_steps") if chunk is None else chunk)
    n_out = getattr(policy, "actions_per_query", 8)  # OFT: an 8-step chunk; original OpenVLA: 1 action per query
    if chunk > n_out:
        raise ValueError(f"open-loop steps {chunk} exceed the {n_out} actions the policy returns per query")
    ins = instruction if instruction is not None else env.instruction
    res = EpisodeResult(env.task_id, init_state, condition, ins)
    extra = {"actions_norm": [], "query_steps": [], "trace": None, "capture": None, "action_info": []}
    if hasattr(action_fn, "reset"):  # per-episode state (vlamech.interventions.mitigation.ensemble)
        action_fn.reset()
    t0 = time.time()
    try:
        frame = env.reset(init_state)
        trace = TraceRecorder(env)
        trace.record()  # row 0: the post-settle frame
        cap = QueryCapture(env, init_state) if capture else None
        queue: list = []
        fed, fresh = None, False
        for step in range(max_steps):
            if not queue:
                masks = mask_fn(frame) if mask_fn is not None else {}
                imgs = {}
                for cam in ("agentview", "wrist"):
                    m = masks.get(cam, {})
                    imgs[cam] = corruptor(cam, getattr(frame, cam), step,
                                          background_mask=m.get("background"), target_mask=m.get("target"))
                views = None
                if corruptor.spec.exact_model_input:  # composite after preprocessing (exact-model-input variant)
                    views = exact_views(frame, imgs, masks, corruptor.spec.cameras, policy.center_crop)
                if cap is not None:  # the exact inputs of this query (raw, what was fed on, masks)
                    cap.record(step, frame, imgs if views is None else views, masks)
                if hasattr(policy, "set_query_context"):  # policies that sample (pi0.5): noise seed of this query
                    policy.set_query_context(env.task_id, init_state, step)
                if action_fn is None:
                    kw = {} if views is None else {"images_224": tuple(Image.fromarray(views[c])
                                                                       for c in ("agentview", "wrist"))}
                    if getattr(policy, "family", "oft") == "pi05":
                        kw["eef_quat"] = frame.eef_quat  # LeRobot computes the state from the quaternion
                    out = policy.forward(policy.prepare(imgs["agentview"], imgs["wrist"], frame.proprio, ins, **kw))
                    acts, norm = out["actions"], out["actions_norm"]
                elif views is not None:
                    raise NotImplementedError("action_fn with exact_model_input")
                else:  # mitigation (vlamech.interventions.mitigation): info is recorded; info["stop"] ends the episode
                    acts, norm, info = action_fn(policy, imgs, frame.proprio, ins)
                    extra["action_info"].append({"step": step, **info})
                    if info.get("stop"):
                        res.stopped_at = step
                        break
                if acts.shape != (n_out, 7):
                    raise RuntimeError(f"policy returned shape {acts.shape}")
                queue = [acts[i] for i in range(chunk)]
                res.n_queries += 1
                if record_actions:
                    extra["actions_norm"].append(np.asarray(norm, np.float32))
                    extra["query_steps"].append(step)
                fed, fresh = (imgs if views is None else views), True
            if frame_hook is not None:
                frame_hook(step, frame, fed, fresh, dict(zip(trace.task_ids, trace.rows["goals"][-1].tolist())))
            fresh = False
            frame, done = env.step(queue.pop(0), raw=getattr(policy, "env_space_actions", False))
            res.steps = step + 1
            trace.record()
            if done:
                res.success = True
                break
        if frame_hook is not None:
            frame_hook(res.steps, frame, None, False, dict(zip(trace.task_ids, trace.rows["goals"][-1].tolist())))
        extra["trace"] = trace.arrays()
        if cap is not None:
            extra["capture"] = cap.arrays()
        summarize(res, extra["trace"], env)
        # consistency check: upstream `done` must agree with our own predicate evaluation
        if res.success != bool(extra["trace"]["goals"][-1, extra["trace"]["goal_task_ids"].tolist().index(env.task_id)]):
            raise RuntimeError("env `done` disagrees with the instructed goal predicate at the last step")
    except Exception:
        res.error = traceback.format_exc()
    res.wall_s = time.time() - t0
    return res, extra


def exact_views(frame: Frame, imgs: dict, masks: dict, cameras, center_crop: bool) -> dict:
    """224x224 network inputs: exact-model-input composite on corrupted cameras, plain preprocessing elsewhere."""
    from vlamech.models.oft import policy_view
    from vlamech.perturbations.leakage import exact_model_input_views

    out = {}
    for cam in ("agentview", "wrist"):
        if cam in cameras:
            out[cam] = exact_model_input_views(getattr(frame, cam), imgs[cam], ~masks[cam]["background"], center_crop)
        else:
            out[cam] = np.asarray(policy_view(imgs[cam], center_crop))
    return out


def summarize(res: EpisodeResult, tr: dict, env: LiberoTaskEnv) -> None:
    """Fill the per-episode summaries from the trace (rows >= 1 are post-action states)."""
    tids = [int(t) for t in tr["goal_task_ids"]]
    g = tr["goals"]
    res.goals_initial = [t for k, t in enumerate(tids) if g[0, k]]
    res.goal_first_step = {t: int(np.argmax(g[1:, k])) for k, t in enumerate(tids) if g[1:, k].any()}
    disp = np.linalg.norm(tr["obj_pos"][1:] - tr["obj_pos"][0], axis=-1)  # (steps, n_obj)
    res.obj_max_disp = {str(n): float(disp[:, i].max(initial=0.0)) for i, n in enumerate(tr["obj_names"])}
    jd = np.abs(tr["joint_q"][1:] - tr["joint_q"][0])
    res.joint_max_disp = {str(n): float(jd[:, i].max(initial=0.0)) for i, n in enumerate(tr["joint_names"])}
    names = [str(n) for n in tr["entity_names"]]
    peak = tr["contact"][1:].max(axis=0, initial=0.0)  # 0 steps (stopped at the first query): no contact
    res.max_contact_force = env.robot_contact_forces(dict(zip(names, peak.tolist())))
    first_off = {}
    for i, n in enumerate(tr["obj_names"]):
        off = [r for r in range(1, len(tr["obj_pos"])) if env.off_table(tr["obj_pos"][r, i])]
        if off:
            first_off[str(n)] = off[0]
    res.off_table = sorted(first_off, key=lambda n: (first_off[n], n))
