"""Run integrity: experiment rules, resolved settings, manifests, resume compatibility, ownership.
CPU only: no model, no simulator."""
import pytest

from vlamech import config
from vlamech.config import split_states
from vlamech.perturbations.image import CorruptionSpec, ladder
from vlamech.runs import experiments
from vlamech.runs import manifest as mf


def _pre(experiment, states, spec=CorruptionSpec("identity"), ins="own", policy="oft"):
    return experiments.preflight(experiment, states, spec, ins, 50, policy=policy)


# ---------------------------------------------------------------------------------------------- experiment rules
def test_split_enforced():
    _pre("baseline", list(range(50)))
    with pytest.raises(ValueError):
        _pre("baseline", split_states("confirmatory"))
    with pytest.raises(ValueError):  # calibration experiment on held-out states
        _pre("severity_calibration", split_states("confirmatory"), CorruptionSpec("noise", ladder("noise")[0]))


def test_calibration_values_must_be_on_the_ladder():
    _pre("severity_calibration", split_states("calibration"), CorruptionSpec("noise", ladder("noise")[2]))
    with pytest.raises(ValueError):
        _pre("severity_calibration", split_states("calibration"), CorruptionSpec("noise", 17.0))


def test_held_out_runs_require_the_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FROZEN_LEVELS", tmp_path / "frozen_levels.yaml")
    with pytest.raises(FileNotFoundError):
        _pre("corruption_sweep", split_states("confirmatory"), CorruptionSpec("noise", ladder("noise")[2]))


def test_instruction_family_and_policy_rules():
    with pytest.raises(ValueError):
        _pre("baseline", list(range(50)), ins="empty")
    with pytest.raises(ValueError):
        _pre("instruction_reliance", list(range(50)), CorruptionSpec("noise", 8.0), ins="empty")
    with pytest.raises(ValueError):
        _pre("baseline", list(range(50)), policy="pi05")
    _pre("model_check", split_states("calibration"), policy="pi05")
    with pytest.raises(ValueError):
        _pre("model_check", split_states("calibration"))


# ---------------------------------------------------------------------------------------------- settings
def test_settings_resolved_and_unsupported_values_rejected(monkeypatch):
    s = experiments.resolve_settings("baseline")
    assert s["env.settle_steps"] == 10 and s["policy.center_crop"] is True
    real = experiments.protocol
    monkeypatch.setattr(experiments, "protocol", lambda k: "float32" if k == "policy.dtype" else real(k))
    with pytest.raises(ValueError):
        experiments.resolve_settings("debug")
    monkeypatch.setattr(experiments, "protocol", lambda k: 20 if k == "env.settle_steps" else real(k))
    with pytest.raises(ValueError):  # breaks comparability outside debug
        experiments.resolve_settings("baseline")
    assert experiments.resolve_settings("debug")["env.settle_steps"] == 20


# ---------------------------------------------------------------------------------------------- manifests
def _manifest(commit="a" * 40, settle=10, asset="x"):
    code = {"project": {"commit": commit, "dirty": False}, "upstream": {}}
    return mf.build_manifest({"experiment": "baseline"}, {"env.settle_steps": settle}, None, {"a": asset}, code,
                             {"repo": "r", "revision": "v"})


def test_identical_manifest_resumes(tmp_path):
    h = mf.open_run(tmp_path / "run", _manifest(), 0, 1)
    h.close()
    h2 = mf.open_run(tmp_path / "run", _manifest(), 0, 1)
    h2.close()
    assert len(list((tmp_path / "run").glob("sessions.jsonl"))) == 1


def test_scientific_change_refuses_resume(tmp_path):
    mf.open_run(tmp_path / "run", _manifest(), 0, 1).close()
    with pytest.raises(RuntimeError, match="different scientific"):
        mf.open_run(tmp_path / "run", _manifest(settle=11), 0, 1)
    with pytest.raises(RuntimeError, match="different scientific"):
        mf.open_run(tmp_path / "run", _manifest(asset="y"), 0, 1)


def test_code_change_needs_recorded_justification(tmp_path):
    mf.open_run(tmp_path / "run", _manifest(), 0, 1).close()
    with pytest.raises(RuntimeError, match="code identity changed"):
        mf.open_run(tmp_path / "run", _manifest(commit="b" * 40), 0, 1)
    h = mf.open_run(tmp_path / "run", _manifest(commit="b" * 40), 0, 1, accept_code_change="refactor, parity tests pass")
    assert h.session["accepted_code_change"] == "refactor, parity tests pass"
    assert h.manifest["code"]["project"]["commit"] == "a" * 40  # the original manifest is preserved
    h.close()


def test_shard_has_single_owner_and_reshard_needs_quiescence(tmp_path):
    h = mf.open_run(tmp_path / "run", _manifest(), 0, 2)
    with pytest.raises(RuntimeError, match="held by another"):
        mf.open_run(tmp_path / "run", _manifest(), 0, 2)
    with pytest.raises(RuntimeError, match="cannot reshard"):
        mf.open_run(tmp_path / "run", _manifest(), 0, 3)
    h.close()
    mf.open_run(tmp_path / "run", _manifest(), 0, 3).close()  # old workers gone: allowed


def test_legacy_run_without_manifest_is_refused(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "episodes.jsonl").write_text('{"task_id": 0, "init_state": 0}\n')
    with pytest.raises(RuntimeError, match="no manifest"):
        mf.open_run(tmp_path / "run", _manifest(), 0, 1)


def test_records_are_stamped(tmp_path):
    h = mf.open_run(tmp_path / "run", _manifest(), 0, 1)
    r = h.stamp({"task_id": 0})
    assert r["manifest_digest"] == h.manifest["digest"] and r["session_id"] == h.session["session_id"]
    h.close()


def test_invalid_shard_rejected(tmp_path):
    for i, n in ((1, 1), (-1, 2), (0, 0)):
        with pytest.raises(ValueError):
            mf.open_run(tmp_path / "run", _manifest(), i, n)


# ---------------------------------------------------------------------------------------------- identity
def test_upstream_identity_matches_pins():
    """Checks the environment the tests run in: OFT's (setup_env.sh) or pi0.5's (setup_env_lerobot.sh)."""
    import importlib.util

    from vlamech.runs import identity

    if importlib.util.find_spec("lerobot") is not None:
        up = identity.upstream_identity(strict=True, policy="pi05")  # raises on any mismatch
        p = identity.pins("pi05")
        assert up["pkg:transformers"]["commit"] == p["PI05_TRANSFORMERS_COMMIT"]
        assert up["pkg:lerobot"]["commit"] == p["LEROBOT_COMMIT"]
        return
    up = identity.upstream_identity(strict=True)  # raises on any mismatch
    assert up["pkg:transformers"]["commit"] == identity.pins()["TRANSFORMERS_FORK_COMMIT"]


def test_experiments_require_the_evidence_later_analyses_need():
    full = {"video", "actions", "captures"}
    for ex in ("clean_capture", "corruption_sweep", "mitigation_test"):
        assert experiments.EXPERIMENTS[ex].records == full, ex  # offline replay, divergence, review videos
    assert {"video", "actions"} <= experiments.EXPERIMENTS["mitigation_selection"].records
    assert "video" in experiments.EXPERIMENTS["severity_calibration"].records


def test_calibration_accepts_exactly_the_rule_extension():
    _pre("severity_calibration", split_states("calibration"), CorruptionSpec("photometric", 4.5))  # 4.0 + 0.5 stops
    with pytest.raises(ValueError):
        _pre("severity_calibration", split_states("calibration"), CorruptionSpec("photometric", 5.0))
    with pytest.raises(ValueError):  # coverage cannot exceed 1: no extension exists
        _pre("severity_calibration", split_states("calibration"), CorruptionSpec("bg_patches", 1.0))


def test_sweep_accepts_exactly_the_frozen_selection(tmp_path, monkeypatch):
    import yaml

    sel = {"corruption": {"identity": {"levels": [None]}, "target_occlusion": {"levels": [None]},
                          "noise": {"levels": [64.0, 128.0]}, "bg_patches": {"levels": [0.8]}},
           "variants": {"exact_model_input": ["bg_patches"]}, "oracle": {"classifier_version": 2}}
    (tmp_path / "frozen_levels.yaml").write_text(yaml.safe_dump(sel))
    monkeypatch.setattr(config, "FROZEN_LEVELS", tmp_path / "frozen_levels.yaml")
    conf = split_states("confirmatory")
    for spec in (CorruptionSpec("identity"), CorruptionSpec("target_occlusion"), CorruptionSpec("noise", 128.0),
                 CorruptionSpec("bg_patches", 0.8, exact_model_input=True)):
        _pre("corruption_sweep", conf, spec)
    for spec in (CorruptionSpec("noise", 32.0), CorruptionSpec("blur", 4.0), CorruptionSpec("matched_bg_box")):
        with pytest.raises(ValueError):
            _pre("corruption_sweep", conf, spec)


def test_condition_labels_name_the_run_directories():
    from vlamech.analysis.arms import spec_of

    assert spec_of("identity").tag == CorruptionSpec("identity").tag
    assert spec_of("blur-8").tag == CorruptionSpec("blur", 8.0).tag
    assert spec_of("bg_patches-0.8").tag == CorruptionSpec("bg_patches", 0.8).tag
    assert spec_of("target_occlusion@wrist").tag == CorruptionSpec("target_occlusion", cameras=("wrist",)).tag
