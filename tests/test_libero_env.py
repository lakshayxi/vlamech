"""LIBERO wrapper: every simulator assumption the oracle and the masks rely on, checked in the simulator.

CPU-only tests (no `sim` marker) check the pure helpers against upstream code.
"""
import numpy as np
import pytest

from vlamech.envs import libero_env as le

PROTECTED_GOAL = {"akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1",
                  "wooden_cabinet_1", "flat_stove_1", "wine_rack_1"}  # read off the 10 BDDL files by hand


# ----------------------------------------------------------------------------- CPU: pure helpers vs upstream
def test_quat2axisangle_matches_upstream_and_does_not_mutate():
    from experiments.robot.libero.libero_utils import quat2axisangle as up

    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q_before = q.copy()
        ours = le.quat2axisangle(q)
        assert np.array_equal(q, q_before)
        np.testing.assert_array_equal(ours, up(q.copy()))
    q = np.array([0.0, 0.0, 0.0, 1.0 + 1e-9])  # upstream clips this in place
    le.quat2axisangle(q)
    assert q[3] == 1.0 + 1e-9


def test_process_policy_action_matches_upstream():
    from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

    rng = np.random.default_rng(1)
    for g in [0.0, 0.2, 0.5, 0.7, 1.0, *rng.uniform(size=20)]:
        a = np.concatenate([rng.normal(size=6), [g]])
        ref = invert_gripper_action(normalize_gripper_action(a.copy(), binarize=True))
        np.testing.assert_array_equal(le.process_policy_action(a), ref)


# ----------------------------------------------------------------------------- simulator


@pytest.mark.sim
def test_entities_derived_from_bddl(goal_env):
    assert set(goal_env.protected_entities) == PROTECTED_GOAL
    assert set(goal_env.own_entities) == {"wooden_cabinet_1"}  # task 0: open the middle drawer
    assert set(goal_env.movable_objects) == {"akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1"}


def _task_id(language: str) -> int:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    ids = [i for i in range(suite.n_tasks) if suite.get_task(i).language == language]
    assert len(ids) == 1
    return ids[0]


@pytest.mark.sim
@pytest.mark.parametrize("language,expected", [
    ("put the bowl on the plate", {"akita_black_bowl_1", "plate_1"}),  # the plate is task-relevant too
    ("open the top drawer and put the bowl inside", {"akita_black_bowl_1", "wooden_cabinet_1"}),
    ("push the plate to the front of the stove", {"plate_1"}),  # target region is on the table (support surface)
])
def test_own_entities_from_bddl(language, expected):
    env = le.LiberoTaskEnv("libero_goal", _task_id(language))
    try:
        assert set(env.own_entities) == expected
    finally:
        env.close()


@pytest.mark.sim
def test_frame_is_read_only(frame0):
    for a in (frame0.agentview, frame0.wrist, frame0.proprio, frame0.agentview_seg):
        assert not a.flags.writeable
    assert frame0.agentview.shape == (256, 256, 3) and frame0.agentview.dtype == np.uint8
    assert frame0.agentview_seg.shape == (256, 256)


@pytest.mark.sim
def test_segmentation_ids_name_the_right_pixels(goal_env, frame0):
    ids = goal_env.instance_ids()
    assert PROTECTED_GOAL <= set(ids), f"missing ids: {PROTECTED_GOAL - set(ids)}"
    seg = frame0.agentview_seg
    for name in ("akita_black_bowl_1", "plate_1", "wooden_cabinet_1", "flat_stove_1"):
        assert (seg == ids[name]).sum() > 20, f"{name} not visible in the third-person view"
    robot_px = np.isin(seg, list(goal_env.robot_segmentation_ids())).sum()
    assert robot_px > 100, "robot ids cover no pixels: id convention is wrong"
    # ids of different entities never overlap the robot range
    assert not set(ids[e] for e in PROTECTED_GOAL) & goal_env.robot_segmentation_ids()


@pytest.mark.sim
def test_masks_partition_and_target_inside_protected(goal_env, frame0):
    m = goal_env.masks(frame0)
    for cam in ("agentview", "wrist"):
        bg, tgt = m[cam]["background"], m[cam]["target"]
        assert not (bg & tgt).any(), "target pixels marked as background"
        assert 0.2 < bg.mean() < 0.98, f"{cam}: implausible background fraction {bg.mean():.2f}"
    assert m["agentview"]["target"].sum() > 20


@pytest.mark.sim
def test_table_geometry_and_off_table_rule(goal_env, frame0):
    t = goal_env.table_top()
    assert abs(t["z"] - 0.90) < 1e-6  # LIBERO tabletop: table_offset z = 0.90 is the top surface
    for name, pos in goal_env.object_positions().items():
        assert not goal_env.off_table(pos), f"{name} flagged off-table at reset"
    below_outside = np.array([t["xy_max"][0] + 0.1, 0.0, t["z"] - 0.2])
    assert goal_env.off_table(below_outside)
    assert not goal_env.off_table(np.array([t["xy_max"][0] + 0.1, 0.0, t["z"] + 0.2]))  # held beyond the edge


@pytest.mark.sim
def test_goal_status_all_false_at_init_and_consistent_with_upstream(goal_env, frame0):
    st = goal_env.goal_status()
    assert set(st) == set(range(10))
    assert not any(st.values()), f"goals already satisfied at init: {st}"
    assert st[goal_env.task_id] == bool(goal_env.inner._check_success())


@pytest.mark.sim
def test_contacts_by_entity(goal_env, frame0):
    robot, owner, entities = goal_env._geom_owners()
    assert robot and not (robot & set(owner))
    assert set(entities) == PROTECTED_GOAL | {"main_table"}
    c = goal_env.robot_contacts()
    assert set(c) == set(entities) | {"other"} and all(v >= 0 for v in c.values())
    f = goal_env.robot_contact_forces(c)
    assert f["object"] == max(c[o] for o in goal_env.movable_objects)
    assert f["environment"] == max(v for k, v in c.items() if k not in goal_env.movable_objects)


# read off the 10 BDDL goal predicates by hand (language order of the suite)
TARGETS = {
    "open the middle drawer of the cabinet": ([], ["wooden_cabinet_1"], ["wooden_cabinet_1_middle_level"]),
    "put the bowl on the stove": (["akita_black_bowl_1"], ["akita_black_bowl_1"], []),
    "put the wine bottle on top of the cabinet": (["wine_bottle_1"], ["wine_bottle_1"], []),
    "open the top drawer and put the bowl inside": (["akita_black_bowl_1"], ["akita_black_bowl_1"],
                                                    ["wooden_cabinet_1_top_level"]),
    "put the bowl on top of the cabinet": (["akita_black_bowl_1"], ["akita_black_bowl_1"], []),
    "push the plate to the front of the stove": (["plate_1"], ["plate_1"], []),
    "put the cream cheese in the bowl": (["cream_cheese_1"], ["cream_cheese_1"], []),
    "turn on the stove": ([], ["flat_stove_1"], ["flat_stove_1_button"]),
    "put the bowl on the plate": (["akita_black_bowl_1"], ["akita_black_bowl_1"], []),
    "put the wine bottle on the rack": (["wine_bottle_1"], ["wine_bottle_1"], []),
}


@pytest.mark.sim
def test_task_targets_derived_from_bddl(goal_env, frame0):
    assert goal_env.articulated_joints == ["flat_stove_1_button", "wooden_cabinet_1_bottom_level",
                                           "wooden_cabinet_1_middle_level", "wooden_cabinet_1_top_level"]
    assert goal_env.joint_positions().shape == (4,)
    for language, (manip, touch, joints) in TARGETS.items():
        t = goal_env.task_targets(_task_id(language))
        assert (t["manipulated_objects"], t["touch_entities"], t["goal_joints"]) == (manip, touch, joints), language
        assert set(t["non_target_joints"]) == set(goal_env.articulated_joints) - set(joints)
        assert not set(t["non_target_objects"]) & set(t["own_entities"])
    six = goal_env.task_targets(_task_id("put the cream cheese in the bowl"))
    assert six["non_target_objects"] == ["plate_1", "wine_bottle_1"]  # the bowl is the destination, not off-target


@pytest.mark.sim
def test_reset_is_deterministic(goal_env):
    a = goal_env.reset(3)
    b = goal_env.reset(3)
    assert np.array_equal(a.agentview, b.agentview) and np.array_equal(a.proprio, b.proprio)


@pytest.mark.sim
def test_reset_reproduces_upstream_episode_sequence_bit_exactly():
    """LIBERO re-samples fixture poses at every reset; upstream's episode k = reset k+1 of a seed-0 env.
    Our reset(k), called in ANY order, must give exactly upstream's episode-k scene and first policy frame."""
    import os

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image, get_libero_wrist_image

    task_id, ks = 4, [0, 1, 2]
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(task_id)
    up = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                            camera_heights=256, camera_widths=256)
    up.seed(0)
    ref = {}
    for k in ks:  # upstream run_episode order and steps
        up.reset()
        obs = up.set_init_state(suite.get_task_init_states(task_id)[k])
        for _ in range(le.SETTLE_STEPS):
            obs, _, _, _ = up.step(get_libero_dummy_action("openvla"))
        ref[k] = (get_libero_image(obs).copy(), get_libero_wrist_image(obs).copy(),
                  np.array(up.env.sim.model.body_pos), up.get_sim_state().copy())
    up.close()
    env = le.LiberoTaskEnv("libero_goal", task_id)
    try:
        for k in [2, 0, 1, 2]:  # scrambled, with a repeat
            f = env.reset(k)
            np.testing.assert_array_equal(np.array(env.inner.sim.model.body_pos), ref[k][2])
            np.testing.assert_array_equal(env.get_sim_state(), ref[k][3])
            np.testing.assert_array_equal(f.agentview, ref[k][0])
            np.testing.assert_array_equal(f.wrist, ref[k][1])
    finally:
        env.close()


@pytest.mark.sim
def test_background_excludes_band_around_protected(goal_env, frame0):
    """Background pixels are never within protected_mask_dilation_px of a protected (or robot) pixel."""
    from scipy import ndimage

    from vlamech.config import protocol

    band = int(protocol("corruption.protected_mask_dilation_px"))
    ids = goal_env.instance_ids()
    prot_ids = [ids[e] for e in goal_env.protected_entities] + list(goal_env.robot_segmentation_ids())
    m = goal_env.masks(frame0)
    for cam, seg in (("agentview", frame0.agentview_seg), ("wrist", frame0.wrist_seg)):
        near = ndimage.binary_dilation(np.isin(seg, prot_ids), structure=np.ones((3, 3), bool), iterations=band)
        assert not (m[cam]["background"] & near).any()
        assert m[cam]["background"].sum() > 0.2 * seg.size


@pytest.mark.sim
def test_segmentation_rendering_changes_neither_frames_nor_physics():
    """Clean captures store masks (for offline re-creation of mask-based conditions): turning segmentation on leaves
    the simulator state and proprio bit-identical and the RGB frames identical up to the renderer's own
    nondeterminism. That floor is measured, not assumed: two identical plain runs occasionally differ by 1 intensity
    level on a single wrist pixel (EGL), so frames may differ by at most 1 level on at most
    a handful of pixels, in either comparison."""
    from vlamech.envs.libero_env import LiberoTaskEnv

    act = np.array([0.4, 0.1, -0.3, 0, 0, 0.1, -1.0])
    out = []
    for seg in (False, True):
        env = LiberoTaskEnv("libero_goal", 8, segmentation=seg)
        frames = [env.reset(3)]
        for _ in range(24):
            frames.append(env.step(act)[0])
        out.append((frames, env.get_sim_state()))
        env.close()
    (fa, sa), (fb, sb) = out
    assert np.array_equal(sa, sb)
    for a, b in zip(fa, fb):
        assert np.array_equal(a.proprio, b.proprio)
        for x, y in ((a.agentview, b.agentview), (a.wrist, b.wrist)):
            d = np.abs(x.astype(int) - y)
            assert d.max() <= 1 and (d > 0).any(-1).sum() <= 5
