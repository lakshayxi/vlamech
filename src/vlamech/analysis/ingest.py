"""Shared ingestion of closed-loop run directories for every analysis.

Used by: scripts/analyze_baseline.py, scripts/analyze_instruction_reliance.py and every later analysis.

load_run(run_dir, expected_keys) reads all episodes*.jsonl files of one run directory and returns a RunData with
explicit accounting: expected / completed / missing / errored-only / conflicting-duplicate keys, plus the
provenance records. `complete` is True only if every expected key has exactly one error-free outcome (identical
re-runs of the same key are tolerated and counted; conflicting outcomes are an error) and every provenance record
(initial + resumes) reports a clean code tree. Analyses must call require_complete() before producing a
completed-study verdict; partial data may only be summarised as an explicitly labelled progress report.

Two run formats: manifest runs (vlamech.runs.manifest: manifest.json + sessions.jsonl; every record must carry the
run's manifest digest, a foreign stamp is an error) and the first baseline and instruction-reliance runs, made
before manifests existed (provenance.json + resumes).
trace(rd, key) loads an episode's oracle trace after checking its SHA-256 against the record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from vlamech.utils.io import read_json, read_jsonl

OUTCOME_FIELDS = ("success", "steps", "goal_first_step")


@dataclass
class RunData:
    run_dir: Path
    expected: set
    records: dict = field(default_factory=dict)  # key -> record (single error-free outcome)
    missing: list = field(default_factory=list)
    errored_only: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    identical_duplicates: int = 0
    unexpected: list = field(default_factory=list)
    provenance: list = field(default_factory=list)
    dirty_sessions: int = 0
    manifest: Optional[dict] = None
    foreign_stamps: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not (self.missing or self.errored_only or self.conflicts or self.unexpected or self.dirty_sessions
                    or self.foreign_stamps)

    def accounting(self) -> dict:
        return {"run_dir": _relative(self.run_dir), "expected": len(self.expected), "completed": len(self.records),
                "missing": len(self.missing), "errored_only": len(self.errored_only), "conflicts": len(self.conflicts),
                "identical_duplicates": self.identical_duplicates, "unexpected": len(self.unexpected),
                "sessions": len(self.provenance), "dirty_sessions": self.dirty_sessions,
                "foreign_stamps": len(self.foreign_stamps),
                "manifest_digest": self.manifest["digest"] if self.manifest else None,
                "commits": sorted({_commit(p) for p in self.provenance} - {None}),
                "complete": self.complete}

    def require_complete(self) -> "RunData":
        if not self.complete:
            raise RuntimeError(f"run is not complete/eligible: {self.accounting()}")
        return self


def grid(tasks: Iterable[int], states: Iterable[int]) -> set:
    return {(t, s) for t in tasks for s in states}


def load_run(run_dir: Path, expected_keys: set) -> RunData:
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("episodes*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no episode files in {run_dir}")
    rd = RunData(run_dir, set(expected_keys))
    if (run_dir / "manifest.json").exists():
        rd.manifest = read_json(run_dir / "manifest.json")
    seen_error = set()
    for f in files:
        for r in read_jsonl(f):
            key = (int(r["task_id"]), int(r["init_state"]))
            if key not in rd.expected:
                rd.unexpected.append(key)
                continue
            if rd.manifest is not None and r.get("manifest_digest") != rd.manifest["digest"]:
                rd.foreign_stamps.append(key)
                continue
            if r.get("error") is not None:
                seen_error.add(key)
                continue
            if key in rd.records:
                prev = rd.records[key]
                if any(prev.get(k) != r.get(k) for k in OUTCOME_FIELDS):
                    rd.conflicts.append(key)
                else:
                    rd.identical_duplicates += 1
                continue
            rd.records[key] = r
    rd.missing = sorted(rd.expected - set(rd.records) - seen_error)
    rd.errored_only = sorted(seen_error - set(rd.records))
    if rd.manifest is not None:
        rd.provenance = read_jsonl(run_dir / "sessions.jsonl")
        rd.dirty_sessions = sum(1 for p in rd.provenance if p["code"]["project"]["dirty"])
    else:
        prov = run_dir / "provenance.json"
        if prov.exists():
            rd.provenance.append(read_json(prov))
        resumes = run_dir / "provenance_resumes.jsonl"
        if resumes.exists():
            rd.provenance += read_jsonl(resumes)
        rd.dirty_sessions = sum(1 for p in rd.provenance if p.get("git", {}).get("dirty"))
    return rd


def _relative(path: Path) -> str:
    """The path relative to the project root when it is inside it (summaries must not record machine paths)."""
    from vlamech.utils.io import PROJECT_ROOT

    try:
        return str(Path(path).absolute().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _commit(p: dict):
    return p["code"]["project"]["commit"] if "code" in p else p.get("git", {}).get("commit")


def trace(rd: RunData, key: tuple) -> dict:
    """The oracle trace of one episode, verified against the SHA-256 in its record."""
    from vlamech.envs.trace import load_trace
    from vlamech.utils.io import file_sha256

    rec = rd.records[key]
    path = rd.run_dir / "traces" / f"task{key[0]}-init{key[1]}.npz"
    if not rec.get("trace_sha256"):
        raise ValueError(f"{key}: record has no trace")
    if file_sha256(path) != rec["trace_sha256"]:
        raise ValueError(f"{path} does not match the hash in its record")
    return load_trace(path)


def find_run(experiment_dir: Path, pattern: str) -> Path:
    runs = sorted(Path(experiment_dir).glob(pattern))
    if len(runs) != 1:
        raise RuntimeError(f"expected exactly one run matching {pattern} in {experiment_dir}, found {runs}")
    return runs[0]


def paired(a: RunData, b: RunData, keys: Optional[set] = None) -> list:
    """Explicit pairing on (task, init_state): every key must be present in both runs (no silent intersection)."""
    keys = set(keys) if keys is not None else (a.expected & b.expected)
    miss = sorted(k for k in keys if k not in a.records or k not in b.records)
    if miss:
        raise RuntimeError(f"{len(miss)} keys lack a record in one of the paired runs, e.g. {miss[:5]}")
    return sorted(keys)
