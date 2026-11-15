"""Replay capture: the exact observations a policy query used, bound to their scene.

Used by: vlamech.envs.episode.run_episode (capture), the open-loop and patching scripts (read), tests/test_trace.py.

A simulator state vector alone does not determine the observation: the fixture layout lives in the MuJoCo model,
not in the state (LIBERO re-samples fixture poses at every reset), and the scene assets can differ. So every captured query stores, per query k:
  step        control step of the query
  agentview, wrist  uint8 camera frames as given to the corruptor (policy orientation, before corruption)
  fed_agentview, fed_wrist  what was actually passed on to the network: the corrupted 256x256 frames, or the
              224x224 network input for the exact-model-input variant (absent for clean conditions: equal to the raw)
  mask_<camera>_<background|target>  bool masks used by mask-dependent corruptions (absent otherwise): offline
              analyses can re-create any corruption at this observation without the simulator
  proprio     float64 (8,)
  sim_state   float64 MuJoCo state (env.get_sim_state()) at the query
and per episode: task_id, init_state, layout_sha256 (fixture layout of this init state, as applied), scene (JSON null:
the benchmark scene), bddl_sha256 (the loaded problem file), schema.
Open-loop analyses use the captured frames directly (no re-rendering): they are the only exact replay.

Why a state vector cannot replay an observation (measured in tests/test_trace.py):
  * robosuite samples observables at the first of the 25 physics substeps of a control step (no delay configured),
    so the frame and proprio a policy receives describe the state ~48 ms before the end-of-step simulator state;
    re-rendering the saved end-of-step state differs (e.g. 0.24 mm proprio, 167 px, already at the post-settle frame)
  * the OSC controller's internal state and the solver warm start are not in the state vector, so re-executing the
    previous action from the previous state diverges as well (2-5 mm)
restore() therefore reproduces exactly only queries at step 0 (env.reset re-creates the post-settle frame
bit-exactly); for later queries it puts the env into the captured physical state and REPORTS the observation
mismatch (strict=True raises). Closed-loop branching from a captured state is approximate and must be labelled so.
"""
from __future__ import annotations

import hashlib

import numpy as np

from vlamech.utils.io import file_sha256, from_json, to_json

CAPTURE_SCHEMA = 1


def layout_sha256(env, init_state: int) -> str:
    lay = env._layout
    h = hashlib.sha256()
    for a in (np.asarray(lay["pos"][init_state], np.float64), np.asarray(lay["quat"][init_state], np.float64)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(to_json(lay["names"]).encode())
    return h.hexdigest()


class QueryCapture:
    def __init__(self, env, init_state: int):
        self.env, self.init_state = env, init_state
        self.rows: dict[str, list] = {k: [] for k in ("step", "agentview", "wrist", "proprio", "sim_state")}

    def record(self, step: int, frame, fed: dict = None, masks: dict = None) -> None:
        self.rows["step"].append(step)
        self.rows["agentview"].append(np.array(frame.agentview, copy=True))
        self.rows["wrist"].append(np.array(frame.wrist, copy=True))
        self.rows["proprio"].append(np.array(frame.proprio, np.float64, copy=True))
        self.rows["sim_state"].append(self.env.get_sim_state())
        for cam in ("agentview", "wrist"):
            if fed is not None:
                self.rows.setdefault(f"fed_{cam}", []).append(np.array(fed[cam], copy=True))
            for kind, m in (masks or {}).get(cam, {}).items():
                if m is not None:
                    self.rows.setdefault(f"mask_{cam}_{kind}", []).append(np.array(m, bool, copy=True))

    def arrays(self) -> dict:
        e = self.env
        n = len(self.rows["step"])
        partial = [k for k, v in self.rows.items() if len(v) not in (0, n)]
        if partial:
            raise RuntimeError(f"capture rows {partial} cover only some queries")
        rows = dict(self.rows)
        for cam in ("agentview", "wrist"):  # fed frames are kept only if they differ from the raw ones somewhere
            fed = rows.pop(f"fed_{cam}", [])
            if any(f.shape != r.shape or not np.array_equal(f, r) for f, r in zip(fed, rows[cam])):
                rows[f"fed_{cam}"] = fed
        out = {k: np.stack(v) for k, v in rows.items() if v}
        out |= {"task_id": np.array(e.task_id), "init_state": np.array(self.init_state),
                "layout_sha256": np.array(layout_sha256(e, self.init_state)), "scene": np.array(to_json(None)),
                "bddl_sha256": np.array(file_sha256(e.bddl_file)), "schema": np.array(CAPTURE_SCHEMA)}
        return out


def load_capture(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        c = {k: z[k] for k in z.files}
    if int(c["schema"]) != CAPTURE_SCHEMA:
        raise ValueError(f"{path}: capture schema {int(c['schema'])}, this code reads {CAPTURE_SCHEMA}")
    for k in ("layout_sha256", "bddl_sha256"):
        c[k] = str(c[k])
    c["scene"] = from_json(str(c["scene"]))
    c["task_id"], c["init_state"] = int(c["task_id"]), int(c["init_state"])
    return c


def restore(env, cap: dict, k: int, strict: bool = True):
    """Put `env` into the state of captured query k. Returns (frame, report); report = {"exact": bool,
    "proprio_max_abs": float, "agentview_px": int, "wrist_px": int}. The binding (task, benchmark scene, problem file,
    fixture layout) must match or it raises; with strict=True a non-identical observation raises too."""
    if env.task_id != cap["task_id"] or cap["scene"] is not None:
        raise ValueError("capture belongs to a different task or scene")
    if file_sha256(env.bddl_file) != cap["bddl_sha256"]:
        raise ValueError("loaded problem file differs from the captured one")
    frame = env.reset(cap["init_state"])  # applies this init state's fixture layout
    if layout_sha256(env, cap["init_state"]) != cap["layout_sha256"]:
        raise ValueError("fixture layout differs from the captured one")
    if int(cap["step"][k]) != 0:
        frame = env.set_sim_state(cap["sim_state"][k])
    report = {"proprio_max_abs": float(np.abs(frame.proprio - cap["proprio"][k]).max()),
              "agentview_px": int(np.any(frame.agentview != cap["agentview"][k], axis=-1).sum()),
              "wrist_px": int(np.any(frame.wrist != cap["wrist"][k], axis=-1).sum())}
    report["exact"] = report["proprio_max_abs"] == 0 and report["agentview_px"] == 0 and report["wrist_px"] == 0
    if strict and not report["exact"]:
        raise RuntimeError(f"restored query {k} does not reproduce the captured observation: {report}")
    return frame, report


def network_inputs(cap: dict, k: int) -> tuple:
    """(agentview, wrist) exactly as passed on to the network at captured query k: the fed frames if stored, else
    the raw ones (clean condition). 224x224 frames are already-preprocessed network inputs (exact-model-input)."""
    return tuple(cap[f"fed_{c}"][k] if f"fed_{c}" in cap else cap[c][k] for c in ("agentview", "wrist"))


def replay_forward(policy, cap: dict, k: int, instruction: str, **kw) -> dict:
    """The policy forward on captured query k (same inputs as during the run). kw go to policy.forward
    (e.g. output_hidden_states=True for patching)."""
    from PIL import Image

    a, w = network_inputs(cap, k)
    if a.shape[:2] == (224, 224):
        inp = policy.prepare(a, w, cap["proprio"][k], instruction, images_224=(Image.fromarray(a), Image.fromarray(w)))
    else:
        inp = policy.prepare(a, w, cap["proprio"][k], instruction)
    return policy.forward(inp, **kw)
