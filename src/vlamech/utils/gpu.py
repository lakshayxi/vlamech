"""Shared-GPU etiquette: check before launching, never interfere with other users.

Used by: every script that loads a model (called at startup).

`require_gpu(min_free_gib)` raises if the GPU does not have enough free memory, and logs every compute
process on the GPU with its owner, so the run's provenance records what else was running.
It never kills or signals any process.
"""
from __future__ import annotations

import getpass
import subprocess


def _query(fields: str, kind: str = "gpu") -> list[list[str]]:
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.check_output(["nvidia-smi", f"{flag}={fields}", "--format=csv,noheader,nounits"], text=True)
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpu_status(index: int = 0) -> dict:
    rows = _query("index,memory.total,memory.used,memory.free,utilization.gpu")
    row = next(r for r in rows if int(r[0]) == index)
    procs = []
    for pid, mem in _query("pid,used_memory", kind="apps"):
        try:
            owner = subprocess.check_output(["ps", "-o", "user=", "-p", pid], text=True).strip()
        except subprocess.CalledProcessError:
            owner = "?"
        procs.append({"pid": int(pid), "used_mib": int(mem), "owner": owner})
    return {"index": index, "total_mib": int(row[1]), "used_mib": int(row[2]), "free_mib": int(row[3]),
            "util_pct": int(row[4]), "processes": procs}


def require_gpu(min_free_gib: float, index: int = 0) -> dict:
    st = gpu_status(index)
    others = [p for p in st["processes"] if p["owner"] != getpass.getuser()]
    msg = (f"GPU{index}: {st['free_mib']/1024:.1f} GiB free of {st['total_mib']/1024:.1f}, util {st['util_pct']}%; "
           f"other users' processes: {others or 'none'}")
    print("[gpu]", msg, flush=True)
    if st["free_mib"] / 1024 < min_free_gib:
        raise RuntimeError(f"not enough free GPU memory (need {min_free_gib} GiB). {msg}. Not starting.")
    return st
