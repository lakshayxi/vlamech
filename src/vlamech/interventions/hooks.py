"""Activation capture and intervention hooks on the OFT language model.

Used by: vlamech.interventions.patching, tests/test_hooks.py.

Sites (all per decoder layer `l`, 0-based):
  "resid"  - output hidden state of decoder layer l (== hidden_states[l+1] in HF output)
  "attn"   - output of the self-attention block of layer l (after o_proj, before the residual add)
  "mlp"    - output of the MLP of layer l (before the residual add)
  "embed"  - the input sequence to layer 0 (== hidden_states[0]); use layer=-1

Interventions are applied at token positions (indices into the full LLM sequence; see TokenLayout).
Kinds:
  replace  - set positions to `value` (patching from another run)        <- observation of mechanism
  add      - add `value` (steering / noise control)
  scale    - multiply by `value` (scalar or broadcastable)

Fail-fast guarantees (tested in tests/test_hooks.py):
  * every registered hook must fire exactly once per forward pass, otherwise RuntimeError
  * tensor shapes are checked against the expected sequence length
  * hooks are always removed on exit (context manager), even if the forward raises
  * tuple / BaseModelOutput structured outputs are handled; the rest of the tuple is passed through
  * at a module with both, interventions run first and captures record the POST-intervention value

What you can change: add sites (e.g. per-head attention outputs via a hook on o_proj input).
Keep the fire-count check: a silently missing hook makes a patching experiment look like a null result.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch

SITES = ("resid", "attn", "mlp", "embed")


@dataclass
class Intervention:
    site: str
    layer: int
    positions: Union[np.ndarray, Sequence[int]]
    kind: str = "replace"  # replace | add | scale
    value: Optional[torch.Tensor] = None  # (len(positions), D) or broadcastable; scalar for scale

    def __post_init__(self):
        if self.site not in SITES:
            raise ValueError(f"unknown site {self.site}")
        if self.kind not in ("replace", "add", "scale"):
            raise ValueError(f"unknown kind {self.kind}")
        if self.site == "embed" and self.layer != -1:
            raise ValueError("embed site uses layer=-1")
        self.positions = torch.as_tensor(np.asarray(self.positions, dtype=np.int64))


@dataclass
class Capture:
    site: str
    layer: int
    positions: Optional[Union[np.ndarray, Sequence[int]]] = None  # None = all positions
    to_cpu: bool = True


def _module_for(policy, site: str, layer: int):
    layers = policy.decoder_layers
    if site == "embed":
        return layers[0], "pre"
    if not 0 <= layer < len(layers):
        raise IndexError(f"layer {layer} out of range")
    if site == "resid":
        return layers[layer], "post"
    if site == "attn":
        return layers[layer].self_attn, "post"
    if site == "mlp":
        return layers[layer].mlp, "post"
    raise AssertionError(site)


def _get_hidden(output):
    """Return (tensor, rebuild_fn) for tensor-or-tuple module outputs."""
    if isinstance(output, torch.Tensor):
        return output, lambda t: t
    if isinstance(output, tuple):
        return output[0], lambda t: (t,) + tuple(output[1:])
    if hasattr(output, "last_hidden_state"):
        def rebuild(t):
            output.last_hidden_state = t
            return output
        return output.last_hidden_state, rebuild
    raise TypeError(f"unsupported module output type {type(output)}")


def _apply(h: torch.Tensor, iv: Intervention) -> torch.Tensor:
    pos = iv.positions.to(h.device)
    h = h.clone()
    if iv.kind == "replace":
        v = iv.value.to(h.device, h.dtype)
        h[:, pos] = v.reshape(h[:, pos].shape) if v.dim() >= 2 else v
    elif iv.kind == "add":
        h[:, pos] = h[:, pos] + iv.value.to(h.device, h.dtype)
    else:
        h[:, pos] = h[:, pos] * (iv.value.to(h.device, h.dtype) if torch.is_tensor(iv.value) else iv.value)
    return h


class HookSession:
    """Holds captured tensors and fire counts for one `hooked()` context."""

    def __init__(self):
        self.captures: dict[tuple, torch.Tensor] = {}
        self.fire_counts: dict[int, int] = {}
        self.expected_seq_len: Optional[int] = None

    def get(self, site: str, layer: int) -> torch.Tensor:
        return self.captures[(site, layer)]


@contextmanager
def hooked(policy, interventions: Sequence[Intervention] = (), captures: Sequence[Capture] = (),
           expected_seq_len: Optional[int] = None, n_forwards: int = 1):
    """Register interventions + captures for the duration of the block.

    `n_forwards` = how many forward passes will run inside the block; each hook must fire that many times.
    Captures keep the value of the *last* forward pass.
    """
    sess = HookSession()
    sess.expected_seq_len = expected_seq_len
    handles = []
    by_module: dict[int, dict] = {}

    def entry(site, layer):
        mod, kind = _module_for(policy, site, layer)
        key = id(mod), kind
        if key not in by_module:
            by_module[key] = {"module": mod, "kind": kind, "ivs": [], "caps": []}
        return by_module[key]

    for iv in interventions:
        entry(iv.site, iv.layer)["ivs"].append(iv)
    for cp in captures:
        entry(cp.site, cp.layer)["caps"].append(cp)

    def make_hook(e, hid):
        def process(h):
            if h.dim() != 3:
                raise RuntimeError(f"expected (B, L, D) activations, got {tuple(h.shape)}")
            if sess.expected_seq_len is not None and h.shape[1] != sess.expected_seq_len:
                raise RuntimeError(f"sequence length {h.shape[1]} != expected {sess.expected_seq_len}")
            for iv in e["ivs"]:
                h = _apply(h, iv)
            for cp in e["caps"]:
                x = h if cp.positions is None else h[:, torch.as_tensor(np.asarray(cp.positions), device=h.device)]
                x = x.detach()
                sess.captures[(cp.site, cp.layer)] = x.to("cpu", copy=True) if cp.to_cpu else x.clone()
            sess.fire_counts[hid] = sess.fire_counts.get(hid, 0) + 1
            return h

        if e["kind"] == "post":
            def post_hook(module, args, output):
                h, rebuild = _get_hidden(output)
                return rebuild(process(h))
            return post_hook

        def pre_hook(module, args, kwargs):
            if args:
                return (process(args[0]),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = process(kwargs["hidden_states"])
            return args, kwargs
        return pre_hook

    try:
        for hid, e in enumerate(by_module.values()):
            fn = make_hook(e, hid)
            if e["kind"] == "post":
                handles.append(e["module"].register_forward_hook(fn))
            else:
                handles.append(e["module"].register_forward_pre_hook(fn, with_kwargs=True))
        yield sess
        missing = {hid: sess.fire_counts.get(hid, 0) for hid in range(len(by_module)) if sess.fire_counts.get(hid, 0) != n_forwards}
        if missing:
            raise RuntimeError(f"hooks fired an unexpected number of times (expected {n_forwards}): {missing}")
    finally:
        for h in handles:
            h.remove()


def capture_all_resid(policy, inp, positions=None, vision_feats=None) -> tuple[dict, dict]:
    """Run one forward and return ({layer: resid tensor on CPU}, policy.forward output dict)."""
    caps = [Capture("resid", l, positions) for l in range(policy.n_layers)]
    with hooked(policy, captures=caps, expected_seq_len=inp.layout.seq_len) as s:
        out = policy.forward(inp, vision_feats=vision_feats)
    return {l: s.get("resid", l) for l in range(policy.n_layers)}, out
