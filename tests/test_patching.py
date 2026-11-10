"""Activation patching: CPU logic (always) and on-model checks (gpu + sim; run before every patching job).

The on-model checks are the ones a silent bug would hide behind:
  * clamping every position at every depth from a clean donor into a corrupted input gives the clean action exactly
    (the patch path itself is complete and exact),
  * a prefix clamp of a group differs from a single-layer patch of it (the prefix really spans layers),
  * the control group never touches a read position.
"""
import numpy as np
import pytest

from vlamech.interventions import patching as P
from vlamech.metrics import open_loop as ol
from vlamech.models.oft import TokenLayout


def layout():
    return TokenLayout(seq_len=20, bos=(0,), agentview=(1, 2), wrist=(3, 4), proprio=(5,), template=(6, 7, 10, 11),
                       instruction=(8, 9), action_slots=tuple(range(12, 19)), readout=tuple(range(11, 18)), stop=(19,))


def test_points_and_groups():
    assert P.points_up_to("embed") == ["embed"]
    assert P.points_up_to(2) == ["embed", 0, 1, 2]
    lay = layout()
    assert list(P.group_positions(lay, "control")) == [6, 7, 10]  # template minus the read position 11
    assert list(P.group_positions(lay, "readout")) == list(range(11, 18))
    assert list(P.group_positions(lay, "vision")) == [1, 2, 3, 4]
    assert P.site_of("embed") == ("embed", -1) and P.site_of(7) == ("resid", 7)


def test_contrast_against_splits_along_and_across_in_float64():
    clean = np.zeros((3, 8, 7), np.float32)
    clean[1, :, 0] = 0.3
    clean[2, :, 1] = 0.7
    same = ol.contrast_against(clean, clean)
    assert same["ier_par"] == pytest.approx(1.0, abs=1e-12)
    assert same["ier_perp"] == pytest.approx(0.0, abs=1e-6)
    assert same["cos_align"] == pytest.approx(1.0, abs=1e-12)
    flipped = ol.contrast_against(-clean, clean)
    assert flipped["ier_par"] == pytest.approx(-1.0, abs=1e-12)


def test_ier_par_against():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(3, 8, 7))
    assert ol.ier_par_against(clean, clean) == pytest.approx(1.0)
    half = clean.copy()
    half[1:] = clean[:1] - 0.5 * (clean[:1] - clean[1:])  # contrasts halved along themselves
    assert ol.ier_par_against(half, clean) == pytest.approx(0.5)
    same = np.repeat(clean[:1], 3, axis=0)  # every instruction gives the own action: no steering at all
    assert ol.ier_par_against(same, clean) == pytest.approx(0.0)


@pytest.mark.gpu
@pytest.mark.sim
class TestOnModel:
    @pytest.fixture(scope="class")
    def pair(self, policy, frame0, goal_env):
        from vlamech.perturbations.image import blur

        clean = policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, goal_env.instruction)
        corr = policy.prepare(blur(frame0.agentview, 8.0), blur(frame0.wrist, 8.0), frame0.proprio, goal_env.instruction)
        return clean, corr

    def test_full_clamp_restores_the_clean_action_exactly(self, policy, pair):
        clean, corr = pair
        caps, out = P.donor(policy, clean)
        pos = np.arange(clean.layout.seq_len)
        got = P.run(policy, corr, P.patches(caps, pos, P.points_up_to(policy.n_layers - 1)))
        np.testing.assert_array_equal(got, out["actions_norm"])

    def test_prefix_spans_layers_and_noise_control_runs(self, policy, pair):
        clean, corr = pair
        cc, _ = P.donor(policy, clean)
        rc, ro = P.donor(policy, corr)
        pos = P.group_positions(corr.layout, "instruction")
        single = P.run(policy, corr, P.patches(cc, pos, [8]))
        prefix = P.run(policy, corr, P.patches(cc, pos, P.points_up_to(8)))
        assert not np.array_equal(single, prefix)
        noisy = P.run(policy, corr, P.noise_patches(cc, rc, pos, P.points_up_to(8), seed=0))
        assert np.isfinite(noisy).all() and not np.array_equal(noisy, ro["actions_norm"])


def test_bowl_states_and_random_donor():
    a = np.ones((5, 8, 7))
    a[2:, :, 6] = 0.0  # gripper closes from chunk 2 on
    assert P.first_closing(a) == 2 and P.bowl_queries(a, 5) == [2, 3] and P.bowl_queries(a, 3) == [2]
    assert P.first_closing(np.ones((4, 8, 7))) is None and P.bowl_queries(np.ones((4, 8, 7)), 4) == []
    states = {"task1-init20": [3, 4], "task1-init21": [5], "task1-init22": [2, 3], "task4-init20": [1, 2]}
    assert P.random_donor_key(states, "task1-init20", 1) == "task1-init22"  # init 21 has no second state
    assert P.random_donor_key(states, "task1-init22", 0) == "task1-init20"  # cyclic
    assert P.random_donor_key(states, "task4-init20", 0) is None  # no other episode of the task
