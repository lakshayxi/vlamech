"""Offline replay of captured queries reproduces the closed-loop actions bit-exactly (GPU): the precondition for
every open-loop and patching result made from captures."""
import numpy as np
import pytest

from vlamech.envs.episode import run_episode
from vlamech.envs.replay import replay_forward
from vlamech.perturbations.image import CorruptionSpec, EpisodeCorruptor


@pytest.mark.gpu
@pytest.mark.sim
@pytest.mark.parametrize("spec", [CorruptionSpec("identity"), CorruptionSpec("noise", 32.0)])
def test_replay_reproduces_recorded_actions(policy, goal_env, spec):
    res, extra = run_episode(goal_env, policy, 2, EpisodeCorruptor(spec, "task0-init2"), max_steps=40, chunk=8,
                             capture=True, record_actions=True)
    assert res.error is None, res.error
    cap = extra["capture"]
    assert cap["step"].tolist() == extra["query_steps"]
    for k in range(len(cap["step"])):
        out = replay_forward(policy, cap, k, goal_env.instruction)
        assert np.array_equal(out["actions_norm"], extra["actions_norm"][k]), f"query {k}"
