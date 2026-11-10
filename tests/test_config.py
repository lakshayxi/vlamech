"""The protocol file is the single source of truth: every parameter must be documented and well-formed."""
import pytest

from vlamech import config


def test_protocol_validates():
    tree = config.protocol_tree()  # raises on any parameter without value/source/why
    assert "corruption" in tree and "statistics" in tree


def test_splits_disjoint_and_cover_all_states():
    cal, exp, conf = (set(config.split_states(n)) for n in ("calibration", "exploratory", "confirmatory"))
    assert not (cal & exp) and not (cal & conf) and not (exp & conf)
    assert cal | exp | conf == set(range(config.protocol("splits.n_init_states")))


def test_upstream_values_match_upstream_code():
    """Values marked `upstream` must match the upstream source they cite."""
    from vlamech.envs.libero_env import SETTLE_STEPS, TASK_MAX_STEPS

    assert config.protocol("env.settle_steps") == SETTLE_STEPS
    assert config.protocol("env.max_steps") == TASK_MAX_STEPS["libero_goal"]


def test_missing_parameter_raises():
    with pytest.raises(KeyError):
        config.protocol("env.does_not_exist")


def test_bad_parameter_rejected():
    with pytest.raises(ValueError):
        config._validate({"x": {"value": 1, "source": "guess", "why": "because"}}, "")
    with pytest.raises(ValueError):
        config._validate({"x": {"value": 1, "source": "rule", "why": "  "}}, "")


def test_frozen_requires_the_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FROZEN_LEVELS", tmp_path / "frozen_levels.yaml")
    with pytest.raises(FileNotFoundError):
        config.frozen("corruption.noise")


def test_hyperparameter_doc_is_current():
    import subprocess, sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts/render_protocol_doc.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
