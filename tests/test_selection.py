"""Severity selection rule (protocol corruption.selection_rule). CPU only."""
import numpy as np
import pytest

from vlamech.metrics.selection import isotonic_nonincreasing, next_level, select


def test_isotonic_pools_violations_with_weights():
    np.testing.assert_allclose(isotonic_nonincreasing([1.0, 0.8, 0.9, 0.2], [1, 1, 1, 1]), [1.0, 0.85, 0.85, 0.2])
    np.testing.assert_allclose(isotonic_nonincreasing([0.5, 0.7], [3, 1]), [0.55, 0.55])
    y = [0.9, 0.95, 0.6, 0.7, 0.1]
    f = isotonic_nonincreasing(y, [50] * 5)
    assert (np.diff(f) <= 1e-12).all()
    with pytest.raises(ValueError):
        isotonic_nonincreasing([1.0, np.nan], [1, 1])


def test_already_monotone_is_unchanged():
    y = [1.0, 0.9, 0.5, 0.1]
    np.testing.assert_allclose(isotonic_nonincreasing(y, [50] * 4), y)


def test_select_closest_and_ties_go_to_the_milder_level():
    out = select([1, 2, 4, 8], [50, 45, 30, 15], [50] * 4, 1.0, [0.9, 0.6, 0.3])
    assert out["chosen"] == {0.9: 2, 0.6: 4, 0.3: 8}
    tie = select([1, 2, 3], [40, 20, 10], [50] * 3, 1.0, [0.6])  # 0.8 and 0.4 are equally far from 0.6
    assert tie["chosen"][0.6] == 1


def test_relative_to_clean_success_and_benign_flag():
    out = select([1, 2], [45, 44], [50, 50], 0.9, [0.9, 0.3])
    np.testing.assert_allclose(out["relative"], [1.0, 44 / 45])
    assert out["top_above_lowest_target"]


def test_ladder_extension_respects_spacing_and_bounds():
    assert next_level([0.5, 1.0, 1.5], "linear", None) == 2.0
    assert next_level([4, 8, 16], "geometric", None) == 32
    assert next_level([0.4, 0.8], "geometric", 1.0) is None  # coverage cannot exceed 1
    assert next_level([3, 1], "geometric", 1) is None  # JPEG quality cannot go below 1


def test_unreachable_targets_and_benign():
    out = select([96, 6, 1], [50, 43, 27], [50] * 3, 1.0, [0.9, 0.6, 0.3])  # JPEG-like: bottom 54%
    assert out["unreachable_targets"] == [0.3] and out["selected_levels"] == [6, 1] and not out["benign"]
    flat = select([0.1, 0.8], [50, 49], [50, 50], 1.0, [0.9, 0.6, 0.3])
    assert flat["benign"] and flat["selected_levels"] == [0.8]
