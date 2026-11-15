"""Open-loop instruction-effect metrics from the actions of scripts/run/instruction_effect.py (protocol metrics.*).

Used by: scripts/analysis/instruction_effect.py, tests/test_open_loop.py.

Input per episode: actions_norm (queries, conditions, instructions, 8, 7) with instruction 0 = own, 1..9 = the other
tasks, last = "" (empty). Arm dims and chunk come from protocol metrics. All distances: Euclidean norm over the
arm dims of the whole 8-step chunk.
  IE(s, c)        mean over the 9 alternative instructions of |a(s,c,own) - a(s,c,alt)|
  log IER(s, c)   log IE(s,c) - log IE(s,clean)                         (protocol metrics.ier_aggregation)
  VE(s, c)        |a(s,c,own) - a(s,clean,own)|
  share(s_k, c)   IE / (IE + VE_state), VE_state = |a(s_k+1,c,own) - a(s_k,c,own)|   (protocol metrics.share_reference)
  task similarity sim(s, c, l') = -|a(s,c,own) - a(s,clean,l')|: which clean task-conditional action the corrupted
                  own-instruction action is closest to (the intent ribbon; exploratory)
Direction of the instruction effect (protocol metrics.ier_decomposition), per alternative l' with the contrast
D(s,c,l') = a(s,c,own) - a(s,c,l') and the clean unit direction u0 = D(s,clean,l') / |D(s,clean,l')|:
  d_par  = <D(s,c,l'), u0>   (signed: negative when corruption reverses the contrast)
  d_perp = sqrt(max(0, |D|^2 - d_par^2)),   cos = d_par / |D|
  IER_par(s,c)  = mean_l' d_par  / IE(s,clean)     (1 on clean; < 1: less change ALONG the clean task contrast)
  IER_perp(s,c) = mean_l' d_perp / IE(s,clean)     (0 on clean; change in other directions)
  cos_align(s,c) = mean_l' cos                     (1 on clean)
  Linear scale (d_par can be negative, d_perp is 0 on clean, so no logs).
Episode values average over the episode's captured states (the episode is the independent unit).
"""
from __future__ import annotations

import numpy as np

from vlamech.config import protocol


def _arm(a: np.ndarray) -> np.ndarray:
    dims = list(protocol("metrics.arm_dims"))
    return a[..., dims].reshape(*a.shape[:-2], -1)  # (..., 8 * len(dims))


def ie(acts: np.ndarray) -> np.ndarray:
    """(queries, conditions) instruction effect from (queries, conditions, instructions, 8, 7)."""
    x = _arm(acts)
    own, alt = x[:, :, :1], x[:, :, 1:-1]
    if alt.shape[2] != 9:
        raise ValueError(f"expected 9 alternative instructions, got {alt.shape[2]}")
    return np.linalg.norm(alt - own, axis=-1).mean(axis=-1)


def log_ier(acts: np.ndarray, clean: int) -> np.ndarray:
    e = ie(acts)
    if (e <= 0).any():
        raise FloatingPointError("instruction effect of exactly 0: log ratio undefined")
    return np.log(e) - np.log(e[:, [clean]])


def decomposition(acts: np.ndarray, clean: int) -> dict:
    """(queries, conditions) IER_par, IER_perp, cos_align: the instruction contrasts split along / across the clean ones."""
    x = _arm(acts)
    d = x[:, :, :1] - x[:, :, 1:-1]  # own - alternative, (queries, conditions, 9, dims)
    if d.shape[2] != 9:
        raise ValueError(f"expected 9 alternative instructions, got {d.shape[2]}")
    n = np.linalg.norm(d, axis=-1)
    n0 = n[:, [clean]]
    if (n0 <= 0).any() or (n <= 0).any():
        raise FloatingPointError("an instruction contrast of exactly 0: its direction is undefined")
    par = (d * (d[:, [clean]] / n0[..., None])).sum(-1)
    perp = np.sqrt(np.maximum(0.0, n**2 - par**2))
    ie0 = n0.mean(-1)
    return {"ier_par": par.mean(-1) / ie0, "ier_perp": perp.mean(-1) / ie0, "cos_align": (par / n).mean(-1)}


def contrast_against(acts: np.ndarray, clean_acts: np.ndarray) -> dict:
    """IER_par, IER_perp, cos_align of `acts` relative to `clean_acts`, for any number of alternatives.

    Both (..., instructions, 8, 7) with instruction 0 = own and the rest the alternatives (no empty instruction). On
    the arm dims, in float64: float32 rounding of a contrast projected on itself gives 1 - 1e-7, which a ratio over
    (1 - IER_par) turns into a blow-up. Returns arrays of shape (...)."""
    x, x0 = _arm(acts).astype(np.float64), _arm(clean_acts).astype(np.float64)
    d, d0 = x[..., :1, :] - x[..., 1:, :], x0[..., :1, :] - x0[..., 1:, :]
    n0 = np.linalg.norm(d0, axis=-1)
    if (n0 <= 0).any():
        raise FloatingPointError("a clean instruction contrast of exactly 0: its direction is undefined")
    n = np.linalg.norm(d, axis=-1)
    par = (d * d0).sum(-1) / n0
    perp = np.sqrt(np.maximum(0.0, n**2 - par**2))
    ie0 = n0.mean(-1)
    return {"ier_par": par.mean(-1) / ie0, "ier_perp": perp.mean(-1) / ie0,
            "cos_align": np.divide(par, n, out=np.zeros_like(par), where=n > 0).mean(-1)}


def ier_par_against(acts: np.ndarray, clean_acts: np.ndarray) -> np.ndarray:
    """IER_par of `contrast_against`: mean over alternatives of <own - alt, clean contrast unit> / mean |clean contrast|;
    1 when acts == clean_acts."""
    return contrast_against(acts, clean_acts)["ier_par"]


def ve(acts: np.ndarray, clean: int) -> np.ndarray:
    x = _arm(acts)[:, :, 0]
    return np.linalg.norm(x - x[:, [clean]], axis=-1)


def share(acts: np.ndarray) -> np.ndarray:
    """(queries - 1, conditions): IE / (IE + VE_state) for every captured state that has a successor."""
    x = _arm(acts)[:, :, 0]
    vs = np.linalg.norm(x[1:] - x[:-1], axis=-1)
    e = ie(acts)[:-1]
    return e / (e + vs)


def episode_values(acts: np.ndarray, clean: int) -> dict:
    """Per-condition episode values: mean log IER, mean VE, mean share difference vs clean, and the means of the
    direction decomposition (IER_par, IER_perp, cos_align; linear scale)."""
    sh = share(acts)
    dec = {k: v.mean(axis=0) for k, v in decomposition(acts, clean).items()}
    return {"log_ier": log_ier(acts, clean).mean(axis=0), "ve": ve(acts, clean).mean(axis=0),
            "share_diff": (sh - sh[:, [clean]]).mean(axis=0) if len(sh) else np.full(acts.shape[1], np.nan), **dec}


def intent_similarity(acts: np.ndarray, clean: int) -> np.ndarray:
    """(queries, conditions, 10): -distance of each condition's own-instruction action to the clean action under each
    of the 10 task instructions (index 0 = own task). argmax = the task the action looks like (intent ribbon)."""
    x = _arm(acts)
    return -np.linalg.norm(x[:, :, :1] - x[:, clean][:, None, :-1], axis=-1)
