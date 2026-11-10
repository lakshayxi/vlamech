"""Exploratory destination label (protocol oracle.destination_label). CPU only, synthetic traces."""
import numpy as np
import pytest

from vlamech.metrics.destination import approach, destinations, heading, incidental, label, shared_objects

TARGETS = {0: {"manipulated_objects": ["bowl"]}, 1: {"manipulated_objects": ["bowl"]},
           2: {"manipulated_objects": ["cheese"]}, 3: {"manipulated_objects": []}}


def trace(path):
    path = np.asarray(path, float)
    other = np.zeros_like(path)
    return {"obj_pos": np.stack([path, other], 1), "obj_names": ["bowl", "cheese"], "targets": TARGETS,
            "goal_task_ids": [0, 1, 2, 3]}


A, B = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
DEST = destinations([(0, trace([[0, 0, 0], A])), (1, trace([[0, 0, 0], B]))])


def test_only_objects_of_several_tasks_have_destinations():
    assert shared_objects(trace([[0, 0, 0]])) == {"bowl": [0, 1]}
    assert set(DEST) == {("bowl", 0), ("bowl", 1)}
    with pytest.raises(ValueError):
        destinations([(0, trace([[0, 0, 0], A]))])  # task 1's destination has no clean success


def test_approach_is_fraction_of_distance_closed():
    assert approach([[0, 0, 0], [0.5, 0, 0], [0.25, 0, 0]], A) == pytest.approx(0.5)


def test_heading_and_wrong_destination():
    h = heading(trace([[0, 0, 0], [0.1, 0.8, 0]]), 0, DEST, progressed=True)  # told task 0 (A), carried toward B
    assert h["heading"] == 1 and label("execution_loss", 0, h) == "wrong_destination"
    assert label("success", 0, h) == "success"
    own = heading(trace([[0, 0, 0], [0.7, 0.1, 0]]), 0, DEST, progressed=True)
    assert own["heading"] == 0 and label("execution_loss", 0, own) == "execution_loss"


def test_not_applicable_without_progress_or_shared_object():
    assert heading(trace([[0, 0, 0], B]), 0, DEST, progressed=False) is None
    assert heading(trace([[0, 0, 0], B]), 2, DEST, progressed=True) is None
    assert label("no_progress", 0, None) == "no_progress"


def test_clean_maximum_filters_incidental_approach():
    # clean successes of task 0 pass 40% of the way toward task 1's destination on their way to A
    inc = incidental([(0, trace([[0, 0, 0], [0.3, 0.4, 0], A])), (1, trace([[0, 0, 0], B]))], DEST)
    assert inc[(0, 1)] == pytest.approx(1 - np.linalg.norm([0.3, -0.6, 0]))
    nudge = trace([[0, 0, 0], [0.0, 0.05, 0]])  # a 5% nudge toward B, nothing toward A
    assert heading(nudge, 0, DEST, True)["heading"] == 1  # plain argmax would call it B
    assert heading(nudge, 0, DEST, True, inc)["heading"] == 0
    far = trace([[0, 0, 0], [0.0, 0.9, 0]])
    assert heading(far, 0, DEST, True, inc)["heading"] == 1
