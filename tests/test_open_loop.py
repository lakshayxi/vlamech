"""Open-loop metrics on synthetic actions with known answers. CPU only."""
import numpy as np
import pytest

from vlamech.metrics import open_loop as ol

Q, C, I = 4, 2, 11  # queries, conditions (0 = clean, 1 = corrupted), instructions (own, 9 others, empty)


def acts(own_clean=0.0, alt_gap=1.0, scale=1.0, drift=0.0):
    a = np.zeros((Q, C, I, 8, 7))
    for q in range(Q):
        for c in range(C):
            s = 1.0 if c == 0 else scale
            a[q, c, 0, :, :6] = (own_clean + drift * q) * s
            for i in range(1, 10):
                a[q, c, i, :, :6] = (own_clean + drift * q + alt_gap * i / 9) * s
            a[q, c, :, :, 6] = 5.0  # gripper differences must never count
    return a


def test_instruction_effect_and_its_ratio():
    a = acts()
    e = ol.ie(a)
    expect = np.mean([i / 9 * np.sqrt(48) for i in range(1, 10)])
    np.testing.assert_allclose(e, expect)
    half = acts(scale=0.5)
    np.testing.assert_allclose(ol.log_ier(half, clean=0)[:, 1], np.log(0.5))
    np.testing.assert_allclose(ol.log_ier(half, clean=0)[:, 0], 0.0)


def test_share_is_invariant_to_global_action_shrinkage():
    a = acts(drift=0.3, scale=0.5)  # corrupted: every action halves, instruction effect and state change alike
    v = ol.episode_values(a, clean=0)
    assert v["share_diff"][1] == pytest.approx(0.0, abs=1e-12)
    assert v["log_ier"][1] == pytest.approx(np.log(0.5))


def test_visual_effect_and_intent_similarity():
    a = acts(drift=0.0)
    a[:, 1, 0, :, :6] = a[:, 0, 5, :, :6]  # corrupted own-instruction action equals clean action of instruction 5
    assert ol.ve(a, clean=0)[0, 1] == pytest.approx(5 / 9 * np.sqrt(48))
    sim = ol.intent_similarity(a, clean=0)
    assert sim.shape == (Q, C, 10) and (sim[:, 1].argmax(-1) == 5).all() and (sim[:, 0].argmax(-1) == 0).all()


def test_zero_instruction_effect_is_refused():
    with pytest.raises(FloatingPointError):
        ol.log_ier(acts(alt_gap=0.0), clean=0)


def contrasts(clean_vec, corrupt_vec):
    """Actions whose own instruction is 0 and whose alternative i (1..9) sits at -(i/9) x vec: the contrast own - alt
    is (i/9) x clean_vec in the clean condition and (i/9) x corrupt_vec in the corrupted one (48-d arm chunk)."""
    a = np.zeros((Q, C, I, 8, 7))
    for c, v in enumerate((clean_vec, corrupt_vec)):
        for i in range(1, 10):
            a[:, c, i, :, :6] = -(i / 9) * np.asarray(v, float).reshape(8, 6)
    return a


E1, E2 = np.eye(48)[0], np.eye(48)[1]


def test_decomposition_on_clean_and_on_a_shrunk_copy():
    d = ol.decomposition(contrasts(E1, 0.5 * E1), clean=0)
    np.testing.assert_allclose(d["ier_par"], [[1.0, 0.5]] * Q)
    np.testing.assert_allclose(d["ier_perp"], [[0.0, 0.0]] * Q, atol=1e-12)
    np.testing.assert_allclose(d["cos_align"], [[1.0, 1.0]] * Q)


def test_decomposition_rotation_and_reversal():
    rot = ol.decomposition(contrasts(E1, E2), clean=0)  # same size, orthogonal direction
    assert rot["ier_par"][0, 1] == pytest.approx(0.0) and rot["ier_perp"][0, 1] == pytest.approx(1.0)
    rev = ol.decomposition(contrasts(E1, -E1), clean=0)  # contrast reversed: negative, no log taken
    assert rev["ier_par"][0, 1] == pytest.approx(-1.0) and rev["cos_align"][0, 1] == pytest.approx(-1.0)


def test_ier_can_rise_while_the_aligned_part_fades():
    """The case the decomposition exists for: 0.8 x the clean contrast plus an equal-size orthogonal part."""
    a = contrasts(E1, 0.8 * E1 + E2)
    assert np.exp(ol.log_ier(a, clean=0)[0, 1]) == pytest.approx(np.sqrt(0.64 + 1.0))  # IER 1.28 > 1
    d = ol.decomposition(a, clean=0)
    assert d["ier_par"][0, 1] == pytest.approx(0.8) and d["ier_perp"][0, 1] == pytest.approx(1.0)
    v = ol.episode_values(a, clean=0)
    assert v["ier_par"][1] == pytest.approx(0.8) and v["cos_align"][1] == pytest.approx(0.8 / np.sqrt(1.64))
