"""OFT wrapper correctness: parity with upstream inference, determinism, layout, attention structure.

These tests guard against the silent errors that would invalidate every downstream experiment.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.sim]


def _upstream_cfg(policy):
    return SimpleNamespace(center_crop=True, num_images_in_input=2, use_proprio=True, unnorm_key=policy.unnorm_key)


def test_matches_upstream_get_vla_action(policy, frame0, goal_env):
    """Our forward must reproduce upstream get_vla_action exactly on a real LIBERO observation."""
    from experiments.robot.openvla_utils import get_vla_action, resize_image_for_policy

    obs = {
        "full_image": resize_image_for_policy(np.array(frame0.agentview), 224),
        "wrist_image": resize_image_for_policy(np.array(frame0.wrist), 224),
        "state": np.array(frame0.proprio, copy=True),
    }
    state_before = obs["state"].copy()
    ref = np.stack(get_vla_action(_upstream_cfg(policy), policy.vla, policy.processor, obs, goal_env.instruction,
                                  action_head=policy.action_head, proprio_projector=policy.proprio_projector))
    # Documented upstream hazard: it overwrites obs["state"] with the normalized proprio.
    assert not np.allclose(obs["state"], state_before), "expected upstream to mutate obs['state'] (hazard check)"

    inp = policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, goal_env.instruction)
    ours = policy.forward(inp)["actions"]
    assert ours.shape == (8, 7)
    np.testing.assert_array_equal(ours, ref)


def test_readout_positions_match_upstream(policy, frame0, goal_env):
    """The 56 hidden states the action head reads must be exactly upstream's actions_hidden_states, i.e. the
    positions one BEFORE each action slot (causal label-shift convention kept by OFT)."""
    inp = policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, goal_env.instruction)
    L = inp.layout
    n_prompt = inp.input_ids.shape[1] - 1
    assert L.readout[0] == 513 + n_prompt == L.action_slots[0] - 1  # NUM_PATCHES + NUM_PROMPT_TOKENS upstream
    attn = torch.ones_like(inp.input_ids)
    with torch.inference_mode():
        _, ref_h = policy.vla.predict_action(
            input_ids=inp.input_ids, attention_mask=attn, pixel_values=inp.pixel_values, unnorm_key=policy.unnorm_key,
            proprio=inp.proprio_norm, proprio_projector=policy.proprio_projector, action_head=policy.action_head,
            do_sample=False)
    ours = policy.forward(inp, output_hidden_states=True)["hidden_states"][-1][:, list(L.readout)]
    assert torch.equal(ours, ref_h)


def test_prepare_does_not_mutate_inputs(policy, frame0):
    a, w, p = np.array(frame0.agentview), np.array(frame0.wrist), np.array(frame0.proprio)
    policy.prepare(a, w, p, "turn on the stove")
    assert np.array_equal(a, frame0.agentview) and np.array_equal(w, frame0.wrist) and np.array_equal(p, frame0.proprio)


def test_deterministic(policy, frame0, goal_env):
    inp = policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, goal_env.instruction)
    a1 = policy.forward(inp)["actions_norm"]
    a2 = policy.forward(inp)["actions_norm"]
    np.testing.assert_array_equal(a1, a2)


def test_cached_vision_features_identical(policy, frame0):
    out = policy.act_many(frame0.agentview, frame0.wrist, frame0.proprio, ["turn on the stove", "put the bowl on the plate"])
    for ins, res in out.items():
        direct = policy.forward(policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, ins))
        np.testing.assert_array_equal(res["actions_norm"], direct["actions_norm"])


def test_layout(policy, frame0):
    ins = "put the cream cheese in the bowl"
    inp = policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, ins)
    L = inp.layout
    assert len(L.agentview) == 256 and len(L.wrist) == 256 and len(L.proprio) == 1 and len(L.action_slots) == 56
    # partition check: every position belongs to exactly one group
    allpos = sorted(sum((list(getattr(L, g)) for g in L.GROUPS), []))
    assert allpos == list(range(L.seq_len))
    # instruction tokens decode back to the task string
    p0 = L.proprio[0] + 1
    toks = inp.input_ids[0, 1:].tolist()
    instr_ids = [toks[p - p0] for p in L.instruction]
    assert policy.tokenizer.decode(instr_ids).strip() == ins


def test_attention_is_bidirectional(policy, frame0):
    """Image-patch hidden states must depend on the (later) instruction tokens. A causal model would not."""
    from vlamech.interventions.hooks import capture_all_resid

    i1 = policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, "turn on the stove")
    i2 = policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, "put the wine bottle on the rack")
    r1, _ = capture_all_resid(policy, i1, positions=i1.layout.agentview)
    r2, _ = capture_all_resid(policy, i2, positions=i2.layout.agentview)
    assert torch.equal(r1[0][:, :0], r2[0][:, :0])  # trivial shape sanity
    diff = (r1[4].float() - r2[4].float()).abs().max().item()
    assert diff > 1e-2, "agentview positions unaffected by instruction: attention is causal, not the OFT fork"


def test_no_eager_attention_anywhere(policy):
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention

    assert all(type(l.self_attn) is LlamaSdpaAttention for l in policy.decoder_layers)
    assert policy.n_layers == 32
