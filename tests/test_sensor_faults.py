"""Closed-loop sensor faults (vlamech.interventions.sensor_faults) with a stand-in policy and simulator."""

from types import SimpleNamespace

import numpy as np
import pytest

from vlamech.interventions.sensor_faults import FREEZE_QUERIES, OFFSET_M, SensorFault


class FakePolicy:
    """Records what it is fed; commands closing from query `close_at` on."""

    def __init__(self, close_at=99):
        self.fed, self.close_at = [], close_at

    def prepare(self, a, w, p, ins):
        return (a, w, np.array(p))

    def forward(self, inp):
        self.fed.append(inp)
        norm = np.zeros((8, 7))
        norm[:, 6] = 0.0 if len(self.fed) - 1 >= self.close_at else 1.0
        return {"actions": norm, "actions_norm": norm}


def _obs(q):
    return {"agentview": np.full((2, 2, 3), q, np.uint8), "wrist": np.full((2, 2, 3), 100 + q, np.uint8)}, np.full(8, float(q))


def _run(fault, n, policy):
    for q in range(n):
        imgs, p = _obs(q)
        fault(policy, imgs, p, "do it")
    return [(int(a[0, 0, 0]), int(w[0, 0, 0]), float(p[0])) for a, w, p in policy.fed]


def _bind(fault, key="task1-init0"):
    fault.env, fault.key = None, key
    fault.offset = np.array([OFFSET_M, 0, 0])


def test_lag_feeds_the_previous_query_of_one_source():
    f = SensorFault("lag_wrist")
    _bind(f)
    fed = _run(f, 3, FakePolicy())
    assert fed == [(0, 100, 0.0), (1, 100, 1.0), (2, 101, 2.0)]
    f = SensorFault("lag_all")
    _bind(f)
    assert _run(f, 3, FakePolicy()) == [(0, 100, 0.0), (0, 100, 0.0), (1, 101, 1.0)]


def test_offset_and_stuck_gripper_touch_only_their_fields():
    f = SensorFault("proprio_offset")
    _bind(f)
    pol = FakePolicy()
    _run(f, 2, pol)
    p = pol.fed[1][2]
    assert np.isclose(p[0], 1 + OFFSET_M) and np.allclose(p[1:], 1.0)
    f = SensorFault("gripper_stuck")
    _bind(f)
    pol = FakePolicy()
    _run(f, 3, pol)
    assert np.allclose(pol.fed[2][2][:6], 2.0) and np.allclose(pol.fed[2][2][6:], 0.0)


def test_grasp_freeze_holds_the_trigger_observation_for_two_queries():
    f = SensorFault("freeze_grasp")
    _bind(f)
    fed = _run(f, 7, FakePolicy(close_at=2))
    # query 2 commands closing -> its observation is fed again at queries 3 and 4
    assert [x[0] for x in fed] == [0, 1, 2, 2, 2, 5, 6]
    assert FREEZE_QUERIES == 2


def test_unknown_fault_is_rejected():
    with pytest.raises(ValueError):
        SensorFault("lag_everything")


@pytest.mark.sim
def test_counterfactual_worlds_render_and_restore(goal_env):
    from vlamech.perturbations import worlds as W

    goal_env.reset(0)
    base = W.World(goal_env.get_sim_state())
    f0 = W.render(goal_env, base)
    moved = W.shift_entity(goal_env, base, "plate_1", np.array([0.05, 0, 0]))
    fixture = W.shift_entity(goal_env, base, "flat_stove_1", np.array([0, 0.05, 0]))
    arm = W.shift_arm(goal_env, base, np.array([0.03, 0, 0]))
    grip = W.flip_gripper(goal_env, base, 0.04)
    frames = [W.render(goal_env, w) for w in (moved, fixture, arm, grip)]
    assert all(np.any(f.agentview != f0.agentview) for f in frames)
    assert np.array_equal(frames[0].proprio, f0.proprio) and np.array_equal(frames[1].proprio, f0.proprio)
    assert np.allclose(frames[2].proprio[:3] - f0.proprio[:3], [0.03, 0, 0], atol=1e-3) and arm.detail["ik_error_m"] < 1e-3
    assert grip.detail["from"] == "open" and frames[3].proprio[6] - frames[3].proprio[7] < 0.01
    assert np.array_equal(W.render(goal_env, base).agentview, f0.agentview)  # model edits are undone


def test_delay_feeds_the_frame_k_steps_back():
    f = SensorFault("delay_2")
    _bind(f)
    pol = FakePolicy()
    for step in range(20):
        imgs, p = _obs(step)
        if step % 8 == 0:
            f(pol, imgs, p, "do it")
        f.observe(step, SimpleNamespace(agentview=imgs["agentview"], wrist=imgs["wrist"], proprio=p), None, False, {})
    assert [int(a[0, 0, 0]) for a, _, _ in pol.fed] == [0, 6, 14]


def test_wait_holds_still_every_other_query_and_feeds_the_frame_from_the_hold():
    f = SensorFault("lag_wrist_wait")
    _bind(f)
    pol = FakePolicy()
    held = []
    for q in range(5):
        imgs, p = _obs(q)
        acts, _, info = f(pol, imgs, p, "do it")
        if info.get("hold"):
            held.append(q)
            assert not acts[:, :6].any() and (acts[:, 6] == 1.0).all()  # no motion, gripper command kept
    assert f.step_factor == 2 and held == [1, 3]
    # queries 2 and 4 see the wrist frame of the hold query before them, the third-person frame of their own
    assert [(int(a[0, 0, 0]), int(w[0, 0, 0])) for a, w, _ in pol.fed] == [(0, 100), (2, 101), (4, 103)]


def test_delay_wait_feeds_the_frame_taken_when_the_hold_began():
    f = SensorFault("delay_8_wait")
    _bind(f)
    pol = FakePolicy()
    for step in range(40):
        imgs, p = _obs(step)
        if step % 8 == 0:
            f(pol, imgs, p, "do it")
        f.observe(step, SimpleNamespace(agentview=imgs["agentview"], wrist=imgs["wrist"], proprio=p), None, False, {})
    assert [int(a[0, 0, 0]) for a, _, _ in pol.fed] == [0, 8, 24]


def test_wait_before_grasp_holds_only_while_the_gripper_is_open():
    f = SensorFault("lag_wrist_waitopen")
    _bind(f)
    pol, kinds = FakePolicy(), []
    for q in range(6):
        imgs, p = _obs(q)
        p[6], p[7] = (0.04, -0.04) if q < 3 else (0.005, -0.005)
        kinds.append("hold" if f(pol, imgs, p, "do it")[2].get("hold") else "query")
    assert kinds == ["query", "hold", "query", "query", "query", "query"]


def test_slow_halves_the_commanded_motion_and_keeps_the_gripper():
    f = SensorFault("lag_wrist_slow")
    _bind(f)
    pol = FakePolicy()
    pol.forward = lambda inp: {"actions": np.full((8, 7), 0.6), "actions_norm": np.ones((8, 7))}
    imgs, p = _obs(0)
    acts = f(pol, imgs, p, "do it")[0]
    assert f.step_factor == 2 and np.allclose(acts[:, :6], 0.3) and np.allclose(acts[:, 6], 0.6)


def test_warp_is_the_identity_when_the_camera_has_not_moved():
    from vlamech.interventions.sensor_faults import warp_to

    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), np.uint8)
    pose = (np.array([0.0, 0.0, 1.3]), np.eye(3), 75.0)
    assert (warp_to(img, pose, pose) == img).all()
