"""Episode table of an open-loop instruction-effect pass (scripts/run/instruction_effect.py output)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vlamech.metrics import open_loop as ol

COLS = ("log_ier", "ve", "share_diff", "ier_par", "ier_perp", "cos_align")


def load(pass_dir: Path) -> pd.DataFrame:
    """One row per (episode, usable condition) with the episode values of vlamech.metrics.open_loop."""
    rows = []
    for f in sorted(pass_dir.glob("task*-init*.npz")):
        z = np.load(f, allow_pickle=False)
        conds, usable = list(z["conditions"]), z["usable"]
        clean = conds.index("identity")
        a = z["actions_norm"][:, usable]
        names = [c for c, u in zip(conds, usable) if u]
        v = ol.episode_values(a, names.index("identity") if usable[clean] else None)
        t, i = (int(x) for x in f.stem.replace("task", "").split("-init"))
        for j, c in enumerate(names):
            rows.append({"task": t, "init_state": i, "condition": c, **{k: v[k][j] for k in COLS},
                         "n_states": a.shape[0]})
    if not rows:
        raise FileNotFoundError(f"no episode files in {pass_dir}")
    return pd.DataFrame(rows)
