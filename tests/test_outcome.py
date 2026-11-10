"""Outcome classifier on synthetic traces. CPU only.

Scene and targets mirror LIBERO-Goal (tests/test_libero_env.py::TARGETS checks them against the BDDL)."""
import numpy as np
import pytest

from vlamech.metrics.outcome import Thresholds, classify

OBJ = ["akita_black_bowl_1", "cream_cheese_1", "plate_1", "wine_bottle_1"]
JOINTS = ["flat_stove_1_button", "wooden_cabinet_1_bottom_level", "wooden_cabinet_1_middle_level",
          "wooden_cabinet_1_top_level"]
ENTS = ["akita_black_bowl_1", "cream_cheese_1", "flat_stove_1", "main_table", "plate_1", "wine_bottle_1",
        "wine_rack_1", "wooden_cabinet_1", "other"]
TARGETS = {
    0: {"manipulated_objects": [], "touch_entities": ["wooden_cabinet_1"], "goal_joints": ["wooden_cabinet_1_middle_level"],
        "non_target_objects": OBJ},
    1: {"manipulated_objects": ["akita_black_bowl_1"], "touch_entities": ["akita_black_bowl_1"], "goal_joints": [],
        "non_target_objects": ["cream_cheese_1", "plate_1", "wine_bottle_1"]},
    7: {"manipulated_objects": [], "touch_entities": ["flat_stove_1"], "goal_joints": ["flat_stove_1_button"],
        "non_target_objects": OBJ},
}
for t in TARGETS.values():
    t["non_target_joints"] = [j for j in JOINTS if j not in t["goal_joints"]]
THR = Thresholds(disp_m=0.01, joint={j: 0.005 for j in JOINTS}, obj_floor={o: 0.0 for o in OBJ},
                 joint_floor={j: 0.0 for j in JOINTS})
R = 100


def trace():
    return {"obj_pos": np.zeros((R, len(OBJ), 3)), "joint_q": np.zeros((R, len(JOINTS))),
            "goals": np.zeros((R, 10), bool), "contact": np.zeros((R, len(ENTS))), "obj_names": OBJ,
            "joint_names": JOINTS, "entity_names": ENTS, "goal_task_ids": list(range(10)), "targets": TARGETS}


def move(tr, name, row, dist=0.05):
    tr["obj_pos"][row:, OBJ.index(name), 0] = dist


def turn(tr, joint, row, q):
    tr["joint_q"][row:, JOINTS.index(joint)] = q


def test_event_order_decides_with_identical_maxima():
    a, b = trace(), trace()
    move(a, "plate_1", 10), move(a, "akita_black_bowl_1", 50)  # off-target first
    move(b, "akita_black_bowl_1", 10), move(b, "plate_1", 50)  # target first
    da = np.linalg.norm(a["obj_pos"] - a["obj_pos"][0], axis=-1).max(0)
    db = np.linalg.norm(b["obj_pos"] - b["obj_pos"][0], axis=-1).max(0)
    assert np.array_equal(da, db)  # same displacement maxima per object
    ca, cb = classify(a, 1, THR), classify(b, 1, THR)
    assert (ca["outcome"], ca["subtype"]) == ("intent_loss", "off_target_first")
    assert (cb["outcome"], cb["subtype"]) == ("execution_loss", "displaced")
    assert cb["flags"]["off_target_after_progress"] is True


def test_drawer_partial_progress_and_wrong_drawer():
    tr = trace()
    turn(tr, "wooden_cabinet_1_middle_level", 30, -0.08)  # half open, goal needs < -0.14
    c = classify(tr, 0, THR)
    assert (c["outcome"], c["subtype"]) == ("execution_loss", "joint")
    tr = trace()
    turn(tr, "wooden_cabinet_1_top_level", 30, -0.08)  # the wrong drawer
    assert classify(tr, 0, THR)["subtype"] == "off_target_first"
    tr = trace()
    tr["contact"][20:, ENTS.index("wooden_cabinet_1")] = 3.0  # touched the cabinet, nothing moved
    assert classify(tr, 0, THR)["subtype"] == "touched"


def test_stove_partial_progress_and_wrong_goal():
    tr = trace()
    turn(tr, "flat_stove_1_button", 40, 0.3)  # below the turn-on range [0.5, 2.1]
    assert classify(tr, 7, THR)["outcome"] == "execution_loss"
    tr = trace()  # instructed: bowl on stove; the stove gets turned on instead
    turn(tr, "flat_stove_1_button", 40, 1.0)
    tr["goals"][41:, 7] = True
    c = classify(tr, 1, THR)
    assert (c["outcome"], c["subtype"]) == ("intent_loss", "wrong_goal")
    assert c["events"]["goal_onset"] == {7: 41}


def test_below_threshold_is_no_progress_and_threshold_is_strict():
    tr = trace()
    move(tr, "plate_1", 10, dist=0.01)  # exactly the threshold: not "more than"
    turn(tr, "wooden_cabinet_1_top_level", 10, 0.005)
    assert classify(tr, 1, THR)["outcome"] == "no_progress"


def test_success_with_incidental_wrong_goal_and_initial_goals():
    tr = trace()
    tr["goals"][:, 5] = True  # already true at the post-settle frame: not an event
    tr["goals"][30:, 7] = True
    tr["goals"][-1, 1] = True
    c = classify(tr, 1, THR)
    assert c["outcome"] == "success" and c["flags"]["incidental_wrong_goal"] == [7]
    assert 5 not in c["events"]["goal_onset"]
    tr = trace()
    tr["goals"][:, 5] = True
    assert classify(tr, 1, THR)["outcome"] == "no_progress"  # initially true goal is no intent loss


def test_goal_that_becomes_true_again_is_an_event():
    tr = trace()
    tr["goals"][:5, 5] = True
    tr["goals"][60:, 5] = True
    assert classify(tr, 1, THR)["events"]["goal_onset"] == {5: 60}


def test_tie_is_not_before():
    tr = trace()
    move(tr, "plate_1", 10), move(tr, "akita_black_bowl_1", 10)
    assert classify(tr, 1, THR)["outcome"] == "execution_loss"


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        Thresholds(float("nan"), {}, {}, {})
    with pytest.raises(ValueError):
        classify(trace(), 1, Thresholds(0.01, {}, {o: 0.0 for o in OBJ}, {}))  # a joint without threshold


def test_off_target_size_is_reported():
    tr = trace()
    move(tr, "plate_1", 10, dist=0.02)
    assert classify(tr, 1, THR)["events"]["off_target_size"] == {"plate_1": 0.02}


def test_v2_target_settling_below_its_floor_is_not_progress():
    """Calibration run photometric 1.5, task 1 init 2: the bowl's own post-settle drift (0.279 um) exceeded the task threshold
    (0.263 um) and counted as 'displaced' at step 3 under v1."""
    thr = Thresholds(disp_m=0.263e-6, joint={j: 0.0 for j in JOINTS},
                     obj_floor={o: 0.28e-6 for o in OBJ}, joint_floor={j: 0.0 for j in JOINTS})
    tr = trace()
    tr["obj_pos"][3:, OBJ.index("akita_black_bowl_1"), 0] = 0.279e-6  # settling only
    assert classify(tr, 1, thr)["outcome"] == "no_progress"
    move(tr, "akita_black_bowl_1", 50, dist=0.02)  # a real move is still seen
    assert classify(tr, 1, thr)["events"]["target_progress"] == 50
