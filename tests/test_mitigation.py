"""Mitigations: CAG, receding horizon, temporal ensembling and the safe stop (vlamech.interventions.mitigation)."""
import numpy as np
import pytest

from vlamech.interventions import mitigation as mit
from vlamech.perturbations.image import CorruptionSpec


class FakePolicy:
    """Normalized chunk depends on the instruction: empty -> 0.1, otherwise 0.1 + effect."""

    def __init__(self, effect=0.2):
        self.effect = effect

    def act_many(self, a, w, p, instructions):
        out = {}
        for ins in instructions:
            n = np.full((8, 7), 0.1) + (self.effect if ins else 0.0)
            out[ins] = {"actions_norm": n, "actions": self.unnormalize(n)}
        return out

    def unnormalize(self, n):
        return 2.0 * np.asarray(n) + 1.0


def test_omega_one_is_the_plain_policy_and_formula_matches_eq4():
    p, imgs = FakePolicy(), {"agentview": None, "wrist": None}
    a1, n1, info = mit.cag(1.0)(p, imgs, None, "put the bowl on the plate")
    plain = p.act_many(None, None, None, ["put the bowl on the plate"])["put the bowl on the plate"]
    np.testing.assert_allclose(n1, plain["actions_norm"])
    np.testing.assert_allclose(a1, plain["actions"])
    a3, n3, info = mit.cag(3.0)(p, imgs, None, "x")
    np.testing.assert_allclose(n3[:, :6], 0.1 + 3.0 * 0.2)  # a(empty) + omega (a(l) - a(empty)), arm dims
    np.testing.assert_allclose(n3[:, 6], 0.3)  # the gripper stays on the own-instruction branch
    assert info["instruction_effect"] == pytest.approx(np.sqrt(48) * 0.2)  # 8x6 arm entries of 0.2
    assert "stop" not in info


def test_safe_stop_flag():
    imgs = {"agentview": None, "wrist": None}
    assert mit.cag(3.0, stop_below=1.0)(FakePolicy(0.01), imgs, None, "x")[2]["stop"] is True
    assert mit.cag(3.0, stop_below=1.0)(FakePolicy(0.5), imgs, None, "x")[2]["stop"] is False
    for bad in (0.5, float("nan")):
        with pytest.raises(ValueError):
            mit.cag(bad)


def test_parse():
    assert mit.parse(None) == (None, None)
    assert mit.parse("cag:3")[1] == {"method": "cag", "omega": 3.0, "safe_stop_below": None}
    assert mit.parse("cag:2:stop0.5")[1]["safe_stop_below"] == 0.5
    assert mit.parse("horizon:4") == (None, {"method": "horizon", "steps": 4, "ensemble": False})
    fn, d = mit.parse("horizon:2:ensemble")
    assert callable(fn) and d == {"method": "horizon", "steps": 2, "ensemble": True}
    assert [mit.label(mit.parse(s)[1]) for s in ("cag:3", "horizon:4", "horizon:2:ensemble")] == ["cag:3", "horizon:4", "horizon:2:ensemble"]
    with pytest.raises(ValueError):
        mit.parse("cfg:3")


def test_experiment_rules(tmp_path, monkeypatch):
    from vlamech import config
    from vlamech.config import split_states
    from vlamech.runs import experiments

    def pre(experiment, states, mitigation=None):
        return experiments.preflight(experiment, states, CorruptionSpec("identity"), "own", 50, mitigation)

    with pytest.raises(ValueError, match="no mitigation"):
        pre("baseline", list(range(50)), {"omega": 3.0, "safe_stop_below": None})
    monkeypatch.setattr(config, "FROZEN_LEVELS", tmp_path / "frozen_levels.yaml")  # no freeze yet
    explo = split_states("exploratory")
    with pytest.raises(ValueError, match="needs --mitigation"):
        pre("mitigation_selection", explo)
    with pytest.raises(ValueError, match="grid"):
        pre("mitigation_selection", explo, {"omega": 2.5, "safe_stop_below": None})
    with pytest.raises(FileNotFoundError):  # omega 1.5 passes the arm check (follow-up); only the missing freeze stops it
        pre("mitigation_selection", explo, {"omega": 1.5, "safe_stop_below": None})
    with pytest.raises(ValueError, match="derived after"):
        pre("mitigation_selection", explo, {"omega": 3.0, "safe_stop_below": 0.1})

def test_cag_leaves_the_gripper_on_the_own_branch():
    class P:
        def act_many(self, a, w, pr, ins):
            own, empty = np.zeros((8, 7)), np.zeros((8, 7))
            own[:, 6], empty[:, 6] = 0.40, 0.05  # both closed; extrapolating them would open the gripper
            return {ins[0]: {"actions_norm": own}, ins[1]: {"actions_norm": empty}}

        def unnormalize(self, n):
            return n

    _, guided, _ = mit.cag(3.0)(P(), {"agentview": None, "wrist": None}, None, "x")
    assert np.all(guided[:, 6] == 0.40)


def test_ensemble_averages_overlapping_predictions_and_keeps_the_newest_gripper():
    class P:
        def __init__(self):
            self.calls = 0

        def act_many(self, a, w, pr, ins):
            self.calls += 1
            act = np.full((8, 7), float(self.calls))  # chunk value = call number
            return {ins[0]: {"actions": act, "actions_norm": act}}

    p, fn = P(), mit.ensemble(4, 0.0)  # m = 0: plain mean
    imgs = {"agentview": None, "wrist": None}
    first, _, _ = fn(p, imgs, None, "x")
    assert np.all(first[:4, :6] == 1.0)
    second, _, _ = fn(p, imgs, None, "x")
    assert np.all(second[:4, :6] == 1.5)  # chunks 1 and 2 both predict steps 4-7 (mean of 1 and 2)
    assert np.all(second[:4, 6] == 2.0)  # the gripper comes from the newest chunk
    fn.reset()
    p.calls = 0
    assert np.all(fn(p, imgs, None, "x")[0][:4, :6] == 1.0)
