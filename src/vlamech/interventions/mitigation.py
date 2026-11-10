"""Training-free mitigations for closed-loop runs (protocol mitigation.*).

Used by: scripts/run_closed_loop.py (--mitigation), vlamech.envs.episode.run_episode (action_fn protocol).

CAG (Fang et al. 2602.17659, Eq. 4, training-free variant, protocol mitigation.cag_variant):
    a = a(o, empty) + omega * (a(o, l) - a(o, empty))      on NORMALIZED actions, arm dims, then unnormalized
  The gripper (dim 6, raw [0, 1], binarized at 0.5) is taken from the own-instruction branch: extrapolating it can turn
  two closed commands into an open one (protocol mitigation.cag_variant).
  The unconditional branch is the empty instruction inside the unchanged prompt template. omega = 1 is the plain
  policy (checked in tests). Guided actions are not clipped (the paper specifies none); the share of guided
  normalized values outside [-1, 1] is recorded per query so its effect can be reported.
Receding horizon: query every k < 8 steps and execute the first k steps of each chunk (run_closed_loop's chunk argument).
Ensemble (Zhao et al. 2023, ACT temporal ensembling): query every k steps; each executed arm step is the exponentially
  weighted mean of every earlier prediction of that step, oldest first (protocol mitigation.ensemble_weight_m); the
  gripper comes from the newest chunk, for the reason given above.
Instruction effect (free by-product of CAG; our RQ3 signal): IE = ||a(o, l) - a(o, empty)||_2 over the normalized
  8x6 arm chunk (protocol metrics.arm_dims), i.e. how much the human's instruction changes what the robot does.
Safe stop (fail-safe containment): if IE at a query falls below a threshold frozen by protocol
  mitigation.safe_stop_rule, the robot stops (the episode ends as a hand-back to a human; never counted as success).

Action-function protocol: action_fn(policy, images, proprio, instruction) -> (actions, actions_norm, info), where
info is a JSON-able dict recorded per query; info["stop"] = True ends the episode before the action is executed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vlamech.config import protocol

ARM = slice(0, 6)


def instruction_effect(norm_l: np.ndarray, norm_empty: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(norm_l)[:, ARM] - np.asarray(norm_empty)[:, ARM]))


def cag(omega: float, stop_below: Optional[float] = None):
    """CAG action function; with stop_below set, also the safe-stop monitor on the instruction effect."""
    if not (np.isfinite(omega) and omega >= 1.0):
        raise ValueError("omega must be finite and >= 1 (1 = no guidance)")
    if stop_below is not None and not (np.isfinite(stop_below) and stop_below >= 0):
        raise ValueError("safe-stop threshold must be finite and >= 0")

    def action_fn(policy, imgs, proprio, instruction):
        out = policy.act_many(imgs["agentview"], imgs["wrist"], proprio, [instruction, ""])
        n_l, n_0 = out[instruction]["actions_norm"], out[""]["actions_norm"]
        guided = np.array(n_l)
        guided[:, ARM] = n_0[:, ARM] + omega * (n_l[:, ARM] - n_0[:, ARM])
        ie = instruction_effect(n_l, n_0)
        info = {"instruction_effect": ie, "out_of_range": float(np.mean(np.abs(guided[:, ARM]) > 1.0))}
        if stop_below is not None:
            info["stop"] = bool(ie < stop_below)
        return policy.unnormalize(guided), guided, info

    return action_fn


def ensemble(steps: int, m: float):
    """Temporal-ensembling action function: query every `steps` steps. Holds per-episode state; run_episode calls reset()."""
    if not 1 <= steps <= 8:
        raise ValueError("steps must be in 1..8")
    history: list = []
    state = {"t": 0}

    def action_fn(policy, imgs, proprio, instruction):
        o = policy.act_many(imgs["agentview"], imgs["wrist"], proprio, [instruction])[instruction]
        acts, norm = np.array(o["actions"]), np.asarray(o["actions_norm"])
        t = state["t"]
        history.append((t, acts.copy()))
        for j in range(steps):
            target = t + j
            preds = np.stack([a[target - s] for s, a in history if s <= target < s + len(a)])
            w = np.exp(-m * np.arange(len(preds)))
            acts[j, ARM] = (w[:, None] * preds[:, ARM]).sum(0) / w.sum()
        state["t"] = t + steps
        while history and history[0][0] + 8 <= state["t"]:
            history.pop(0)
        return acts, norm, {}

    def reset():
        history.clear()
        state["t"] = 0

    action_fn.reset = reset
    return action_fn


def label(desc: dict) -> str:
    """The protocol name of a mitigation description, e.g. cag:3, horizon:4, horizon:2:ensemble."""
    if desc.get("method", "cag") == "cag":
        return f"cag:{desc['omega']:g}"
    return f"horizon:{desc['steps']}" + (":ensemble" if desc["ensemble"] else "")


def parse(spec: Optional[str]):
    """'cag:<omega>[:stop<threshold>]' | 'horizon:<k>[:ensemble]' -> (action_fn or None, description); None -> (None, None)."""
    if not spec:
        return None, None
    parts = spec.split(":")
    if parts[0] == "horizon" and len(parts) in (2, 3) and (len(parts) == 2 or parts[2] == "ensemble"):
        steps, ens = int(parts[1]), len(parts) == 3
        if not 1 <= steps <= 8:
            raise ValueError(f"bad --mitigation {spec!r}: steps must be in 1..8")
        fn = ensemble(steps, float(protocol("mitigation.ensemble_weight_m"))) if ens else None
        return fn, {"method": "horizon", "steps": steps, "ensemble": ens}
    if parts[0] != "cag" or len(parts) not in (2, 3):
        raise ValueError(f"bad --mitigation {spec!r}; expected cag:<omega>[:stop<threshold>] or horizon:<k>[:ensemble]")
    omega = float(parts[1])
    stop = None
    if len(parts) == 3:
        if not parts[2].startswith("stop"):
            raise ValueError(f"bad safe-stop spec in {spec!r}")
        stop = float(parts[2][len("stop"):])
    return cag(omega, stop), {"method": "cag", "omega": omega, "safe_stop_below": stop}
