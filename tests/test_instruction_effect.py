"""Offline re-creation equals what the closed loop fed (simulator, scripted policy; no network)."""
import numpy as np
import pytest

from vlamech.envs import offline
from vlamech.envs.episode import run_episode
from vlamech.perturbations.image import CorruptionSpec, EpisodeCorruptor


class Scripted:
    center_crop = True

    def prepare(self, *a, **k):
        return None

    def forward(self, _):
        a = np.tile(np.array([0.5, 0.1, -0.3, 0, 0, 0, 0.0]), (8, 1))
        return {"actions": a, "actions_norm": a}


@pytest.mark.sim
@pytest.mark.parametrize("spec", [CorruptionSpec("bg_patches", 0.4), CorruptionSpec("noise", 64.0),
                                  CorruptionSpec("bg_patches", 0.4, exact_model_input=True),
                                  CorruptionSpec("target_occlusion")])
def test_recreate_equals_closed_loop_input(goal_env, spec):
    sides = [20, 30, 40]
    key = "task0-init1"
    _, extra = run_episode(goal_env, Scripted(), 1, EpisodeCorruptor(spec, key, sides), max_steps=24, chunk=8,
                           capture=True, mask_fn=goal_env.masks)
    cap = extra["capture"]
    corr = EpisodeCorruptor(spec, key, sides)
    for k in range(len(cap["step"])):
        av, wr, v224 = offline.recreate(corr, cap, k, center_crop=True)
        if v224 is None:
            assert np.array_equal(av, cap.get("fed_agentview", cap["agentview"])[k])
            assert np.array_equal(wr, cap.get("fed_wrist", cap["wrist"])[k])
        else:
            assert np.array_equal(np.asarray(v224[0]), cap["fed_agentview"][k])
            assert np.array_equal(np.asarray(v224[1]), cap["fed_wrist"][k])


def test_instruction_set_is_own_then_others_then_empty():
    class Suite:
        n_tasks = 3

        def get_task(self, t):
            return type("T", (), {"language": f"task {t}"})()

    assert offline.instructions_for(1, Suite()) == ["task 1", "task 0", "task 2", ""]


@pytest.mark.sim
def test_recreate_on_a_clean_capture_matches_the_corrupted_closed_loop_first_query(goal_env):
    """At the first query the clean and the corrupted episode are in the same state (same init, settle), so the open-loop pass's
    re-creation from a CLEAN capture equals what the corrupted closed loop fed at step 0."""
    sides, key, spec = [20, 30, 40], "task0-init1", CorruptionSpec("bg_patches", 0.4)
    _, clean = run_episode(goal_env, Scripted(), 1, EpisodeCorruptor(CorruptionSpec("identity"), key), max_steps=8,
                           chunk=8, capture=True, mask_fn=goal_env.masks)
    _, bad = run_episode(goal_env, Scripted(), 1, EpisodeCorruptor(spec, key, sides), max_steps=8, chunk=8,
                         capture=True, mask_fn=goal_env.masks)
    av, wr, _ = offline.recreate(EpisodeCorruptor(spec, key, sides), clean["capture"], 0, center_crop=True)
    d = np.abs(av.astype(int) - bad["capture"]["fed_agentview"][0])
    assert d.max() <= 1 and (d > 0).any(-1).sum() <= 5  # renderer floor: EGL can differ by one level on a wrist pixel
