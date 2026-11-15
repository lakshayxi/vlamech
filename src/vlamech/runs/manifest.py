"""Immutable run manifests, resume compatibility and single-writer guarantees.

A run directory results/<experiment>/<name>/ holds
  manifest.json    written once: everything that determines the results (request, resolved settings, frozen
                   selection, protocol/asset hashes, verified code / upstream / model identity) and its digest
  sessions.jsonl   one record per process that wrote to the run (time, host, shard, code identity, digest)
  topology.json    the shard count; changed only while no worker of the old topology holds its lock
  .locks/          flock files: init (manifest creation/comparison) and one per shard (held for the process life)
Resuming requires an identical manifest digest. The only permitted difference is the code section, and only with
an explicit, recorded justification (--accept-code-change "<why>"), for example a refactor verified by the tests.
Every episode record carries the manifest digest and the session id.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import os
import platform
import uuid
from pathlib import Path
from typing import Optional

from vlamech.config import FROZEN_LEVELS
from vlamech.utils.io import (PROJECT_ROOT, append_jsonl, atomic_write_json, file_sha256, read_json, read_jsonl,
                              to_json)

SCHEMA = 1


def digest(obj) -> str:
    return hashlib.sha256(to_json(obj).encode()).hexdigest()


def scientific_digest(manifest: dict) -> str:
    return digest({k: v for k, v in manifest.items() if k not in ("code", "created", "digest", "scientific_digest")})


def build_manifest(request: dict, settings: dict, frozen: Optional[dict], assets: dict, code: dict, model: dict) -> dict:
    m = {
        "schema": SCHEMA,
        "request": request,  # experiment, suite, tasks, states, condition, instruction, replicate, records
        "settings": settings,
        "frozen": frozen,
        "protocol_sha256": file_sha256(PROJECT_ROOT / "configs" / "protocol.yaml"),
        "frozen_levels_sha256": file_sha256(FROZEN_LEVELS) if frozen else None,
        "assets": assets,
        "model": model,
        "code": code,
    }
    m["scientific_digest"] = scientific_digest(m)
    m["digest"] = digest({k: v for k, v in m.items() if k != "digest"})
    return m


class _Lock:
    def __init__(self, path: Path, blocking: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a+")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            self.f.close()
            raise RuntimeError(f"{path} is held by another live process") from None

    def release(self):
        if not self.f.closed:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()


class RunHandle:
    """Owns one shard of a run for the lifetime of the process."""

    def __init__(self, run_dir: Path, manifest: dict, shard_i: int, shard_n: int, session: dict, lock: _Lock):
        self.run_dir, self.manifest, self.shard_i, self.shard_n = run_dir, manifest, shard_i, shard_n
        self.session, self._lock = session, lock

    @property
    def log(self) -> Path:
        return self.run_dir / ("episodes.jsonl" if self.shard_n == 1 else f"episodes.shard{self.shard_i}of{self.shard_n}.jsonl")

    def done_keys(self) -> set:
        """Keys finished in any shard file (safe: other shards of the current topology only add their own keys)."""
        return {(r["task_id"], r["init_state"]) for f in sorted(self.run_dir.glob("episodes*.jsonl"))
                for r in read_jsonl(f) if r.get("error") is None}

    def stamp(self, record: dict) -> dict:
        return record | {"manifest_digest": self.manifest["digest"], "session_id": self.session["session_id"]}

    def close(self):
        self._lock.release()


def open_run(run_dir: Path, manifest: dict, shard_i: int, shard_n: int, accept_code_change: Optional[str] = None) -> RunHandle:
    if not (shard_n >= 1 and 0 <= shard_i < shard_n):
        raise ValueError(f"invalid shard {shard_i}/{shard_n}")
    run_dir = Path(run_dir)
    locks = run_dir / ".locks"
    init = _Lock(locks / "init.lock", blocking=True)
    try:
        mpath = run_dir / "manifest.json"
        if mpath.exists():
            old = read_json(mpath)
            if old.get("schema") != SCHEMA:
                raise RuntimeError(f"{run_dir} has manifest schema {old.get('schema')}, this code writes {SCHEMA}")
            if old["digest"] != manifest["digest"]:
                if old["scientific_digest"] != manifest["scientific_digest"]:
                    raise RuntimeError(f"{run_dir} was created with different scientific settings/assets "
                                       f"({old['scientific_digest'][:12]} vs {manifest['scientific_digest'][:12]}); "
                                       "start a new run instead of mixing results")
                if not accept_code_change:
                    raise RuntimeError(f"{run_dir}: code identity changed ({old['code']['project']['commit'][:10]} -> "
                                       f"{manifest['code']['project']['commit'][:10]}); pass --accept-code-change "
                                       "'<why this cannot change results>' to resume, or start a new run")
        elif any(run_dir.glob("episodes*.jsonl")):
            raise RuntimeError(f"{run_dir} has episodes but no manifest: it cannot be resumed")
        else:
            manifest = manifest | {"created": _dt.datetime.now().isoformat()}
            atomic_write_json(mpath, manifest)
            old = manifest
        # topology: may change only when no worker of the old topology is alive
        tpath = run_dir / "topology.json"
        if tpath.exists():
            old_n = read_json(tpath)["n"]
            if old_n != shard_n:
                held = []
                for i in range(old_n):
                    try:
                        _Lock(locks / f"shard-{i}-of-{old_n}.lock", blocking=False).release()
                    except RuntimeError:
                        held.append(i)
                if held:
                    raise RuntimeError(f"cannot reshard {old_n}->{shard_n}: shards {held} of the old topology are running")
                atomic_write_json(tpath, {"n": shard_n})
        else:
            atomic_write_json(tpath, {"n": shard_n})
        shard_lock = _Lock(locks / f"shard-{shard_i}-of-{shard_n}.lock", blocking=False)
    finally:
        init.release()
    session = {"session_id": uuid.uuid4().hex[:12], "time": _dt.datetime.now().isoformat(), "host": platform.node(),
               "pid": os.getpid(), "shard": [shard_i, shard_n], "manifest_digest": old["digest"],
               "code": manifest["code"], "accepted_code_change": accept_code_change}
    append_jsonl(run_dir / "sessions.jsonl", session)
    return RunHandle(run_dir, old, shard_i, shard_n, session, shard_lock)
