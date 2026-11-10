"""Atomic artifact writes, JSON I/O and code identity."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _umask_mode() -> int:
    """Permissions a normally created file would get (mkstemp would otherwise leave artifacts at 0600)."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def _json_default(obj: Any):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


# Upstream experiments/robot/openvla_utils.py calls json_numpy.patch(), which REPLACES json.dumps/loads/dump/load
# for the whole process (ndarrays become base64 blobs, lists may come back as arrays). Every experiment process
# imports that module, so this file never calls those functions: it uses the encoder/decoder classes, which the
# patch does not touch (tests/test_io.py checks this after importing upstream).
def to_json(obj: Any, indent: Optional[int] = None) -> str:
    """Strict JSON: NaN/Infinity raise (undefined values must be written explicitly, e.g. as null)."""
    return json.JSONEncoder(default=_json_default, indent=indent, allow_nan=False).encode(obj)


def from_json(text: str) -> Any:
    return json.JSONDecoder().decode(text)


def read_json(path: Path) -> Any:
    return from_json(Path(path).read_text())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory, then rename (atomic on POSIX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, _umask_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, to_json(obj, indent=2).encode())


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.chmod(tmp, _umask_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line and fsync. Each line is a complete record, so a crash loses at most one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(to_json(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _scan_jsonl(path: Path) -> tuple[list, str]:
    """(records, tail_state) where tail_state is 'ok' (empty or newline-terminated), 'unterminated' (a complete
    record without its newline: a crash between the closing brace and the newline) or 'truncated' (an incomplete
    final record: a crash mid-write). A malformed NEWLINE-TERMINATED line can only be corruption: it raises."""
    text = path.read_text()
    parts = text.split("\n")
    body, tail = parts[:-1], parts[-1]  # tail == "" iff the file ends with a newline
    out = []
    for i, line in enumerate(body):
        if not line.strip():
            continue
        try:
            out.append(from_json(line))
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"{path}: malformed newline-terminated line {i + 1}: {e.msg}", e.doc, e.pos)
    if not tail.strip():
        return out, "ok"
    try:
        out.append(from_json(tail))
        return out, "unterminated"
    except json.JSONDecodeError:
        return out, "truncated"


def read_jsonl(path: Path) -> list[dict]:
    """All complete records. A truncated final record (crash mid-append) is dropped with a warning; a complete but
    unterminated final record is kept; a malformed newline-terminated line raises."""
    path = Path(path)
    if not path.exists():
        return []
    out, tail = _scan_jsonl(path)
    if tail == "truncated":
        print(f"[io] WARNING: dropping truncated last record of {path}", file=sys.stderr, flush=True)
    return out


def repair_jsonl(path: Path) -> int:
    """Make `path` safe to append to (call before resuming, by the file's only writer): terminate a complete
    unterminated last record, drop a truncated one. Returns the number of records dropped (0 or 1)."""
    path = Path(path)
    if not path.exists():
        return 0
    out, tail = _scan_jsonl(path)
    if tail == "unterminated":
        with open(path, "a") as f:
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        return 0
    if tail == "truncated":
        atomic_write_bytes(path, "".join(to_json(r) + "\n" for r in out).encode())
        return 1
    return 0


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    # "dirty" = uncommitted changes to code / configs / docs. results/ is excluded: it holds outputs, and runs in
    # progress write there, which must not block other runs. Changes there are reported separately.
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", ".", ":(exclude)results")),
            "results_changes": bool(run("status", "--porcelain", "--", "results"))}


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
