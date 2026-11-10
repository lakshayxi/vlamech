"""Failure-monitor signals and the split-conformal threshold (vlamech.metrics.monitor)."""
import math

import numpy as np

from vlamech.metrics import monitor as M


def chunks():
    a = np.zeros((2, 8, 7))
    a[0, :, 0] = 1.0  # first chunk moves along x at speed 1
    a[1, :, 1] = 0.5  # second moves along y at speed 0.5
    a[1, -1, 1] = 0.25
    return a


def test_signals_use_only_the_previous_chunk():
    s = M.query_signals(chunks(), np.array([0.1, 0.2]))
    assert math.isnan(s["chunk_turn"][0]) and math.isnan(s["chunk_jump"][0])
    assert s["chunk_turn"][1] == 90.0  # x then y
    assert np.isclose(s["chunk_jump"][1], np.sqrt(1 + 0.25))
    assert np.isclose(s["horizon_decay"][1], np.log(2.0))  # slows from 0.5 to 0.25
    assert s["horizon_decay"][0] == 0.0 and np.allclose(s["instruction_effect"], [0.1, 0.2])


def test_conformal_threshold_rank():
    scores = np.arange(1.0, 20.0)  # N = 19, alpha 0.05: rank ceil(20 * 0.95) = 19
    assert M.conformal_threshold(scores, 0.05) == 19.0
    assert M.conformal_threshold(scores[:18], 0.05) == float("inf")  # N = 18: rank 19 exceeds N


def test_average_precision():
    assert M.average_precision(np.array([3.0, 2.0, 1.0]), np.array([1, 1, 0])) == 1.0
    assert np.isclose(M.average_precision(np.array([3.0, 2.0, 1.0]), np.array([0, 1, 1])), (1 / 2 + 2 / 3) / 2)


def test_first_exceedance_ignores_nan():
    assert M.first_exceedance(np.array([np.nan, 0.1, 2.0, 3.0]), 1.0) == 2
    assert M.first_exceedance(np.array([np.nan, 0.1]), 1.0) is None
