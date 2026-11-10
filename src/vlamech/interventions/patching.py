"""Activation patching on OFT: donor captures and patch construction (protocol patching.*).

Used by: scripts/run/activation_patching.py, scripts/run/patching_rollouts.py, tests/test_patching.py.

Points on the depth axis (protocol patching.layer_grid): "embed" = the input sequence to layer 0 (site "embed"), an
integer l = the residual output of decoder layer l (site "resid"). Patching at a point replaces the token group's
state there with the donor's; a PREFIX clamp replaces it at every point up to and including that point, so later
mixing cannot re-contaminate the group before the clamp ends (attention is bidirectional).
Token groups are positions (TokenLayout): after layer 0 every position carries information mixed in from every
other, so a group means "these positions' states", not a modality's pure content.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

from vlamech.interventions.hooks import Capture, Intervention, hooked

Point = Union[str, int]


def site_of(point: Point) -> tuple:
    return ("embed", -1) if point == "embed" else ("resid", int(point))


def points_up_to(point: Point) -> list:
    """Every depth point from the input through `point` (all layers, not only the grid's): the prefix of a clamp."""
    return ["embed"] if point == "embed" else ["embed", *range(int(point) + 1)]


def group_positions(layout, group: str) -> np.ndarray:
    """Positions of a token group; `control` = template positions that the action head does not read."""
    if group == "control":
        read = set(layout.readout)
        return np.array([p for p in layout.template if p not in read])
    if group == "readout":
        return np.array(layout.readout)
    return layout.group(group)


@torch.inference_mode()
def donor(policy, inp, vision_feats=None) -> tuple:
    """One forward that keeps every depth point's full-sequence state on the device: ({(site, layer): tensor}, out)."""
    caps = [Capture("embed", -1, to_cpu=False)] + [Capture("resid", l, to_cpu=False) for l in range(policy.n_layers)]
    with hooked(policy, captures=caps, expected_seq_len=inp.layout.seq_len) as s:
        out = policy.forward(inp, vision_feats=vision_feats)
    return dict(s.captures), out


def patches(donor_caps: dict, positions: np.ndarray, points: Sequence[Point]) -> list:
    """Replace `positions` at every point in `points` with the donor's state there."""
    pos = torch.as_tensor(np.asarray(positions, dtype=np.int64))
    out = []
    for p in points:
        site, layer = site_of(p)
        out.append(Intervention(site, layer, positions, "replace", donor_caps[(site, layer)][:, pos.to(
            donor_caps[(site, layer)].device)]))
    return out


def noise_patches(donor_caps: dict, recipient_caps: dict, positions: np.ndarray, points: Sequence[Point],
                  seed: int) -> list:
    """Control: add Gaussian noise whose norm (over the group, per point) equals that of the donor - recipient
    difference there, instead of the donor's state."""
    pos = np.asarray(positions, dtype=np.int64)
    gen = torch.Generator().manual_seed(seed)
    out = []
    for p in points:
        site, layer = site_of(p)
        delta = (donor_caps[(site, layer)][:, pos] - recipient_caps[(site, layer)][:, pos]).float()
        z = torch.randn(delta.shape, generator=gen).to(delta.device)
        z = z * (delta.norm() / z.norm().clamp_min(1e-12))
        out.append(Intervention(site, layer, positions, "add", z))
    return out


@torch.inference_mode()
def run(policy, inp, interventions: Sequence[Intervention], vision_feats=None) -> np.ndarray:
    """Normalized action chunk (8, 7) of one forward with the interventions applied."""
    with hooked(policy, interventions=interventions, expected_seq_len=inp.layout.seq_len):
        out = policy.forward(inp, vision_feats=vision_feats)
    return out["actions_norm"]



# ---------------------------------------------------------------------------------------------- states and donors
GRIPPER_CLOSED_BELOW = 0.5  # raw [0, 1] gripper output; upstream binarizes at 0.5


def first_closing(actions_norm: np.ndarray):
    """Index of the first chunk whose commanded gripper output has chunk mean < 0.5, or None."""
    closed = actions_norm[:, :, 6].mean(-1) < GRIPPER_CLOSED_BELOW
    return int(np.argmax(closed)) if closed.any() else None


def bowl_queries(actions_norm: np.ndarray, n_queries: int) -> list:
    """Captured queries at offsets 0 and +1 from the first closing chunk (protocol patching.bowl_states)."""
    g = first_closing(actions_norm)
    return [] if g is None else [k for k in (g, g + 1) if k < n_queries]


def bowl_instructions(task: int, suite, bowl_tasks: Sequence[int]) -> list:
    """The instructed bowl task first, then the other bowl tasks in task order (protocol patching.instructions_bowl)."""
    return [suite.get_task(task).language] + [suite.get_task(t).language for t in sorted(bowl_tasks) if t != task]


def random_donor_key(states: dict, key: str, offset_index: int):
    """Next episode of the same task (cyclic, init-state order) with a state at the same offset (protocol
    patching.random_donor). `states` maps episode keys 'task<t>-init<i>' to their selected queries."""
    t = key.split("-")[0]
    same = sorted((k for k in states if k.split("-")[0] == t), key=lambda k: int(k.split("init")[1]))
    i = same.index(key)
    for j in range(1, len(same)):
        cand = same[(i + j) % len(same)]
        if len(states[cand]) > offset_index:
            return cand
    return None
