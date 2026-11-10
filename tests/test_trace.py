"""Oracle traces from real episodes: the emitted records contain everything the classifier needs.
Simulator only: a scripted policy replaces the network."""
import numpy as np
import pytest

from vlamech.envs.episode import run_episode
from vlamech.envs.trace import TRACE_SCHEMA, load_trace
from vlamech.perturbations.image import CorruptionSpec, EpisodeCorruptor
from vlamech.utils.io import atomic_write_npz


class Scripted:
    """Moves the end effector forward and down, gripper open: touches the scene within a few dozen steps."""

    def prepare(self, *a):
        return None

    def forward(self, _):
        a = np.tile(np.array([0.6, 0.0, -0.4, 0, 0, 0, 0.0]), (8, 1))
        return {"actions": a, "actions_norm": a}


def _run(env, steps, hook=None):
    return run_episode(env, Scripted(), 0, EpisodeCorruptor(CorruptionSpec("identity"), "t"), max_steps=steps,
                       chunk=8, frame_hook=hook)


@pytest.mark.sim
def test_trace_content_and_summaries(goal_env, tmp_path):
    res, extra = _run(goal_env, 40)
    assert res.error is None, res.error
    tr = extra["trace"]
    R = res.steps + 1  # row 0 = post-settle frame
    assert tr["obj_pos"].shape == (R, 4, 3) and tr["joint_q"].shape == (R, 4) and tr["goals"].shape == (R, 10)
    assert tr["contact"].shape == (R, len(tr["entity_names"])) and tr["eef_pos"].shape == (R, 3)
    assert not tr["goals"][0].any() and res.goals_initial == []
    assert np.isfinite(tr["obj_pos"]).all() and (tr["contact"] >= 0).all()
    disp = np.linalg.norm(tr["obj_pos"] - tr["obj_pos"][0], axis=-1)[1:].max(0)
    assert np.allclose(disp, [res.obj_max_disp[n] for n in tr["obj_names"]], rtol=0, atol=0)
    atomic_write_npz(tmp_path / "t.npz", **tr)
    back = load_trace(tmp_path / "t.npz")
    assert int(back["schema"]) == TRACE_SCHEMA and set(back["targets"]) == set(range(10))
    assert back["targets"][0] == goal_env.task_targets(0)


@pytest.mark.sim
def test_recording_and_hook_do_not_change_the_simulation(goal_env):
    seen = []
    goal_env.reset(0)  # bare loop: no oracle reads at all
    for _ in range(24):
        goal_env.step(Scripted().forward(None)["actions"][0])
    bare = goal_env.get_sim_state()
    _run(goal_env, 24, hook=lambda *x: seen.append(x[0]))
    assert np.array_equal(bare, goal_env.get_sim_state())
    assert seen == list(range(24)) + [24]  # before every action, and once after the last step


@pytest.mark.sim
def test_trace_schema_mismatch_refused(goal_env, tmp_path):
    _, extra = _run(goal_env, 8)
    atomic_write_npz(tmp_path / "t.npz", **(extra["trace"] | {"schema": np.array(TRACE_SCHEMA + 1)}))
    with pytest.raises(ValueError, match="schema"):
        load_trace(tmp_path / "t.npz")


# ----------------------------------------------------------------------------- replay capture (roadmap B)
@pytest.mark.sim
def test_capture_holds_the_policy_inputs_and_documents_replay_limits(goal_env, tmp_path):
    from vlamech.envs.replay import load_capture, restore

    fed = []
    hook = lambda step, frame, f, fresh, goals: fed.append((step, frame.agentview.copy(), frame.proprio.copy())) if fresh else None
    res, extra = run_episode(goal_env, Scripted(), 2, EpisodeCorruptor(CorruptionSpec("identity"), "t"), max_steps=40,
                             chunk=8, capture=True, frame_hook=hook)
    assert res.error is None, res.error
    atomic_write_npz(tmp_path / "c.npz", **extra["capture"])
    cap = load_capture(tmp_path / "c.npz")
    assert cap["step"].tolist() == [s for s, _, _ in fed] == [0, 8, 16, 24, 32] and cap["init_state"] == 2
    for k, (_, img, prop) in enumerate(fed):  # the capture IS what the policy was given
        assert np.array_equal(cap["agentview"][k], img) and np.array_equal(cap["proprio"][k], prop)
    goal_env.reset(7)
    restore(goal_env, cap, 0)  # step 0: exact (reset re-creates the post-settle frame)
    with pytest.raises(RuntimeError, match="does not reproduce"):  # M7: later observations are not re-renderable
        restore(goal_env, cap, 2)
    _, rep = restore(goal_env, cap, 2, strict=False)
    assert not rep["exact"] and np.array_equal(goal_env.get_sim_state(), cap["sim_state"][2])


@pytest.mark.sim
def test_capture_refuses_a_different_scene_binding(goal_env, tmp_path):
    from vlamech.envs.replay import restore

    _, extra = run_episode(goal_env, Scripted(), 2, EpisodeCorruptor(CorruptionSpec("identity"), "t"), max_steps=8,
                           chunk=8, capture=True)
    cap = dict(extra["capture"])
    cap |= {"task_id": 0, "init_state": 2, "scene": None, "bddl_sha256": str(cap["bddl_sha256"]),
            "layout_sha256": "0" * 64}
    with pytest.raises(ValueError, match="layout"):
        restore(goal_env, cap, 0)


@pytest.mark.sim
def test_capture_recreates_the_corrupted_input_offline(goal_env, tmp_path):
    """Captured masks + the deterministic corruptor re-create, without the simulator, exactly what was fed on."""
    from vlamech.envs.replay import load_capture, network_inputs

    spec = CorruptionSpec("bg_patches", 0.4)
    sides = [20, 30, 40]
    res, extra = run_episode(goal_env, Scripted(), 1, EpisodeCorruptor(spec, "task0-init1", sides), max_steps=24,
                             chunk=8, capture=True, mask_fn=goal_env.masks)
    assert res.error is None, res.error
    atomic_write_npz(tmp_path / "c.npz", **extra["capture"])
    cap = load_capture(tmp_path / "c.npz")
    assert {"fed_agentview", "fed_wrist", "mask_agentview_background", "mask_wrist_target"} <= set(cap)
    redo = EpisodeCorruptor(spec, "task0-init1", sides)
    for k, step in enumerate(cap["step"]):
        for i, cam in enumerate(("agentview", "wrist")):
            img = redo(cam, cap[cam][k], int(step), background_mask=cap[f"mask_{cam}_background"][k],
                       target_mask=cap[f"mask_{cam}_target"][k])
            assert np.array_equal(img, network_inputs(cap, k)[i])


@pytest.mark.sim
def test_clean_capture_stores_no_duplicate_frames(goal_env):
    _, extra = run_episode(goal_env, Scripted(), 1, EpisodeCorruptor(CorruptionSpec("identity"), "t"), max_steps=8,
                           chunk=8, capture=True)
    assert "fed_agentview" not in extra["capture"] and "mask_agentview_background" not in extra["capture"]
