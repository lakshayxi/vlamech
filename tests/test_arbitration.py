"""Shares of the policy's response to a changed world (vlamech.analysis.arbitration)."""
import numpy as np
import pytest

from vlamech.analysis import arbitration as A


def _actions(points):
    """(2**n, 8, 7) chunks whose flattened arm dims start with the given 2-d points (rest zero)."""
    out = np.zeros((len(points), 8, 7))
    for i, p in enumerate(points):
        out[i, 0, :2] = p
    return out


def test_subsets_and_index_agree():
    for n in (1, 2, 3):
        subs = A.subsets(n)
        assert len(subs) == 2 ** n and subs[0] == (0,) * n and subs[-1] == (1,) * n
        assert [A.index(b) for b in subs] == list(range(2 ** n))


def test_additive_response_gives_each_source_its_part():
    # source 0 moves the action by (1, 0), source 1 by (2, 0): shares 1/3 and 2/3
    acts = _actions([(0, 0), (1, 0), (2, 0), (3, 0)])
    v, delta = A.values(acts)
    phi = A.shapley(v, 2)
    assert np.allclose(phi, [1 / 3, 2 / 3]) and np.isclose(phi.sum(), 1)
    s = A.single_source(acts, 1)
    assert np.isclose(s["v"], 2 / 3) and np.isclose(s["residual2"], 0)


def test_one_source_decides():
    # only source 2 (proprio) matters; three sources
    pts = [(4 * ((c >> 2) & 1), 0) for c in range(8)]
    phi = A.shapley(A.values(_actions(pts))[0], 3)
    assert np.allclose(phi, [0, 0, 1])


@pytest.mark.parametrize("seed", range(5))
def test_shapley_sums_to_one_with_interactions(seed):
    rng = np.random.default_rng(seed)
    acts = rng.normal(size=(8, 8, 7))
    phi = A.shapley(A.values(acts)[0], 3)
    assert np.isclose(phi.sum(), 1)


def test_off_axis_residual():
    # the conflicting input moves the action sideways, off the line between the two worlds
    acts = _actions([(0, 0), (0, 1), (0, 0), (2, 0)])
    s = A.single_source(acts, 0)
    assert np.isclose(s["v"], 0) and np.isclose(s["residual2"], 1) and np.isclose(s["delta2"], 4)


def test_weighted_share_and_gripper():
    phi = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]])
    assert np.allclose(A.weighted_share(phi, np.array([3.0, 1.0, 5.0])), [0.75, 0.25])
    acts = np.zeros((2, 8, 7))
    acts[1, :, 6] = 1
    assert A.gripper_closed(acts).tolist() == [True, False]
