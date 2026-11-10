"""pi0.5 (LeRobot) wrapper: input conventions, noise seeding, policy settings (vlamech.models.pi05).

CPU tests run in both environments where they can; tests needing lerobot skip in the OFT environment; GPU tests
(marked gpu) run in the LeRobot environment (env/setup_env_lerobot.sh) with the pinned checkpoint.
"""
import numpy as np
import pytest

from vlamech.models import pi05


def _frames(seed=0, size=256):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (size, size, 3), dtype=np.uint8), rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def test_raw_observation_undoes_our_rotation():
    a, w = _frames()
    obs = pi05.raw_observation(a, w, [0.1, 0.2, 0.3], [0, 0, 0, 1], [0.04, -0.04])
    assert obs["pixels"]["image"].shape == (1, 256, 256, 3) and obs["pixels"]["image"].dtype == np.uint8
    np.testing.assert_array_equal(obs["pixels"]["image"][0], a[::-1, ::-1])
    np.testing.assert_array_equal(obs["pixels"]["image2"][0], w[::-1, ::-1])
    assert obs["robot_state"]["eef"]["quat"].shape == (1, 4) and obs["robot_state"]["eef"]["pos"].dtype == np.float64
    with pytest.raises(TypeError):
        pi05.raw_observation(a.astype(np.float32), w, [0, 0, 0], [0, 0, 0, 1], [0, 0])


def test_noise_is_seeded_and_future_zero_is_the_closed_loop_draw():
    s = pi05.noise_seed("rep0", 3, 21, 50)
    assert s == pi05.noise_seed("rep0", 3, 21, 50)
    assert s != pi05.noise_seed("rep0", 3, 21, 100) and s != pi05.noise_seed("rep1", 3, 21, 50)
    one, four = pi05.noise(s, 1, 50, 32), pi05.noise(s, 4, 50, 32)
    assert one.shape == (1, 50, 32) and four.shape == (4, 50, 32)
    assert np.array_equal(one[0].numpy(), four[0].numpy())  # fan of futures: future 0 = closed-loop action
    assert not np.array_equal(four[0].numpy(), four[1].numpy())
    assert abs(float(four.mean())) < 0.05 and abs(float(four.std()) - 1) < 0.05


def test_policy_settings_replace_only_policy_keys():
    from vlamech.runs.experiments import EXPERIMENTS, resolve_settings

    base = resolve_settings("model_check")
    assert resolve_settings("model_check", "oft") == base
    s = resolve_settings("model_check", "pi05")
    assert s["policy.checkpoint"] == pi05.REPO_ID and s["policy.family"] == "pi05" and s["policy.name"] == "pi05"
    assert s["env.open_loop_steps"] == 10 and base["env.open_loop_steps"] == 8 and s["policy.chunk_size"] == 50
    assert "policy.center_crop" not in s and s["policy.global_seed"] == base["policy.global_seed"]
    assert {k: v for k, v in s.items() if k.startswith("env.") and k != "env.open_loop_steps"} == \
        {k: v for k, v in base.items() if k.startswith("env.") and k != "env.open_loop_steps"}
    assert "pi05" not in EXPERIMENTS["corruption_sweep"].policies and EXPERIMENTS["model_check"].split == "calibration"
    with pytest.raises(ValueError):
        resolve_settings("model_check", "nope")


def test_every_protocol_policy_is_selectable():
    from vlamech.config import protocol_tree
    from vlamech.runs.experiments import POLICIES

    assert set(POLICIES) == {"oft"} | set(protocol_tree()["policies"])


def test_identity_pins_per_policy():
    from vlamech.runs.identity import pins

    p = pins("pi05")
    assert {"LEROBOT_COMMIT", "PI05_TRANSFORMERS_COMMIT", "LIBERO_COMMIT"} <= set(p)
    assert "PI05_TRANSFORMERS_COMMIT" not in pins("oft") and "TRANSFORMERS_FORK_COMMIT" in pins("oft")


def test_libero_processor_state_and_images_match_our_frames():
    pytest.importorskip("lerobot")
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.utils.constants import OBS_STATE
    from robosuite.utils.transform_utils import quat2axisangle

    a, w = _frames(1)
    rng = np.random.default_rng(2)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    q *= np.sign(q[3])
    pos, grip = np.array([0.05, -0.2, 1.1]), np.array([0.039, -0.038])
    obs = preprocess_observation(pi05.raw_observation(a, w, pos, q, grip))
    obs["task"] = ["x"]
    out = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])(obs)
    img = (out["observation.images.image"][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    np.testing.assert_array_equal(img, a)  # LeRobot's flip cancels the un-rotation: the network sees our frame
    state = out[OBS_STATE][0].numpy()
    ours = np.concatenate([pos, quat2axisangle(q.copy()), grip]).astype(np.float32)
    np.testing.assert_allclose(state, ours, atol=2e-6)  # float32 (LeRobot) vs float64 (robosuite) axis-angle


def test_view_is_the_224_network_image():
    pytest.importorskip("lerobot")
    a, _ = _frames(3)
    v = pi05.Pi05Policy.view(a)
    assert v.shape == (224, 224, 3) and v.dtype == np.uint8
    flat = np.full((256, 256, 3), 77, np.uint8)
    assert (pi05.Pi05Policy.view(flat) == 77).all()  # resize, no padding for a square frame


@pytest.fixture(scope="module")
def pi05_policy():
    pytest.importorskip("lerobot")
    from vlamech.config import model_revision
    from vlamech.utils.gpu import require_gpu

    try:
        rev = model_revision(pi05.REPO_ID)
    except KeyError:
        pytest.skip(f"{pi05.REPO_ID} not pinned in configs/models.lock.yaml yet")
    require_gpu(min_free_gib=20)
    return pi05.Pi05Policy(pi05.REPO_ID, revision=rev)


@pytest.mark.gpu
def test_weights_fully_loaded(pi05_policy):
    """LeRobot's loader reports embed_tokens as missing: it is tied to lm_head (checked here, with every tensor)."""
    import torch
    from safetensors import safe_open

    pg = pi05_policy.model.model.paligemma_with_expert.paligemma
    assert pg.model.language_model.embed_tokens.weight is pg.lm_head.weight
    sd = pi05_policy.model.state_dict()
    with safe_open(str(pi05_policy.path / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        assert all(torch.equal(sd[k].cpu().to(f.get_tensor(k).dtype), f.get_tensor(k)) for k in keys)
    assert [k for k in sd if k not in keys] == ["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"]


@pytest.mark.gpu
def test_forward_is_deterministic_and_sample_zero_matches(pi05_policy, goal_env):
    p = pi05_policy
    f = goal_env.reset(0)
    p.set_query_context(goal_env.task_id, 0, 0)
    inp = p.prepare(f.agentview, f.wrist, f.proprio, goal_env.instruction, eef_quat=f.eef_quat)
    a1, a2 = p.forward(inp), p.forward(inp)
    assert a1["actions"].shape == (10, 7) and np.array_equal(a1["actions"], a2["actions"])
    fan = p.sample(inp, 3)
    assert fan["actions"].shape == (3, 50, 7)
    assert np.array_equal(fan["actions"][0, :10], a1["actions"])  # future 0 IS the closed-loop action
    assert not np.array_equal(fan["actions"][0], fan["actions"][1])
    fast = p.sample(inp, 3, batched=True)["actions"]  # batch-size-dependent bf16 kernels: close, not identical
    np.testing.assert_allclose(fast[0, :10], a1["actions"], atol=1e-2)
    assert np.abs(a1["actions"][:, :6]).max() < 5 and np.abs(a1["actions"][:, 6]).max() <= 1.5
    p.set_query_context(goal_env.task_id, 0, 50)
    assert not np.array_equal(p.forward(p.prepare(f.agentview, f.wrist, f.proprio, goal_env.instruction,
                                                  eef_quat=f.eef_quat))["actions"], a1["actions"])
