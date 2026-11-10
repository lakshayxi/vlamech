"""Artifact I/O and seeding: crash safety, provenance, refusal to mix results, seed stability."""
import json

import numpy as np
import pytest

from vlamech.utils import io
from vlamech.utils.seeding import derive_seed


def test_atomic_json_roundtrip_with_numpy(tmp_path):
    p = tmp_path / "a" / "b.json"
    io.atomic_write_json(p, {"x": np.int64(3), "y": np.float32(0.5), "z": np.arange(3), "b": np.bool_(True)})
    assert io.read_json(p) == {"x": 3, "y": 0.5, "z": [0, 1, 2], "b": True}
    assert not list(p.parent.glob(".*.tmp")), "temp file left behind"


def test_jsonl_truncated_last_line_dropped_and_repaired(tmp_path):
    p = tmp_path / "r.jsonl"
    io.append_jsonl(p, {"i": 0})
    io.append_jsonl(p, {"i": 1})
    with open(p, "a") as f:
        f.write('{"i": 2, "trunc')  # simulated crash mid-append
    assert [r["i"] for r in io.read_jsonl(p)] == [0, 1]
    assert io.repair_jsonl(p) == 1
    io.append_jsonl(p, {"i": 2})
    assert [r["i"] for r in io.read_jsonl(p)] == [0, 1, 2]


def test_jsonl_corruption_in_the_middle_raises(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"i": 0}\nnot json\n{"i": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        io.read_jsonl(p)


def test_derive_seed_is_stable_golden_value():
    """Changing derive_seed would silently change every stimulus; this pins it."""
    assert derive_seed("vlamech-v1", "noise", "task3-init27", "agentview", 5) == 2200975121
    assert derive_seed("a", 1) != derive_seed("a", 2)
    assert derive_seed("a", "1") == derive_seed("a", 1)  # keys are joined as strings (documented behavior)


def test_io_immune_to_upstream_json_numpy_patch(tmp_path):
    """Upstream openvla_utils monkey-patches the json module on import; our artifacts must stay plain JSON."""
    import experiments.robot.openvla_utils  # noqa: F401  (applies json_numpy.patch())

    p = tmp_path / "r.jsonl"
    io.append_jsonl(p, {"a": np.arange(3), "b": np.float64(1.5)})
    text = p.read_text()
    assert "__numpy__" not in text and '"a": [0, 1, 2]' in text
    rec = io.read_jsonl(p)[0]
    assert isinstance(rec["a"], list) and rec["b"] == 1.5


def test_jsonl_unterminated_complete_record_survives_append(tmp_path):
    """Crash between a record's closing brace and its newline: the record must not be lost."""
    p = tmp_path / "r.jsonl"
    p.write_text('{"i": 1}')
    assert io.read_jsonl(p) == [{"i": 1}]
    assert io.repair_jsonl(p) == 0
    io.append_jsonl(p, {"i": 2})
    assert [r["i"] for r in io.read_jsonl(p)] == [1, 2]


def test_jsonl_malformed_terminated_last_line_raises(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"i": 1}\nnot json\n')
    with pytest.raises(json.JSONDecodeError):
        io.read_jsonl(p)


def test_strict_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        io.atomic_write_json(tmp_path / "x.json", {"v": float("nan")})
