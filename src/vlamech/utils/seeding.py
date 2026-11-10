"""Seeding and determinism controls."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    # (PYTHONHASHSEED is not set here: it only takes effect at interpreter start. No study code depends on
    # hash() ordering; stimulus randomness uses derive_seed below.)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except ImportError:
        pass


def derive_seed(*keys) -> int:
    """Stable 32-bit seed from arbitrary keys (e.g. namespace, family, episode key, camera, step).

    Uses SHA-256 rather than Python's hash() so it is identical across processes and runs.
    """
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return int.from_bytes(h[:4], "little")

