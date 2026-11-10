"""Hook correctness: hooks must not change ordinary inference, must capture the right tensors,
identity patches must be exact, and missing/misaligned hooks must fail loudly."""
import numpy as np
import pytest
import torch

from vlamech.interventions.hooks import Capture, Intervention, capture_all_resid, hooked

pytestmark = [pytest.mark.gpu, pytest.mark.sim]


@pytest.fixture(scope="module")
def inp(policy, frame0, goal_env):
    return policy.prepare(frame0.agentview, frame0.wrist, frame0.proprio, goal_env.instruction)


def test_capture_does_not_change_output(policy, inp):
    base = policy.forward(inp)["actions_norm"]
    caps, out = capture_all_resid(policy, inp)
    np.testing.assert_array_equal(out["actions_norm"], base)
    assert len(caps) == 32 and caps[0].shape == (1, inp.layout.seq_len, 4096)


def test_capture_matches_hf_hidden_states(policy, inp):
    out = policy.forward(inp, output_hidden_states=True)
    caps, _ = capture_all_resid(policy, inp)
    for l in (0, 7, 15, 31):
        # HF hidden_states[l+1] is the output of layer l, except the last entry, which is after the final norm
        if l == 31:
            continue
        assert torch.equal(caps[l], out["hidden_states"][l + 1].cpu())


@pytest.mark.parametrize("site", ["resid", "attn", "mlp"])
def test_identity_patch_exact(policy, inp, site):
    base = policy.forward(inp)["actions_norm"]
    pos = inp.layout.group("instruction")
    with hooked(policy, captures=[Capture(site, 10, pos)]) as s:
        policy.forward(inp)
    val = s.get(site, 10)
    with hooked(policy, interventions=[Intervention(site, 10, pos, "replace", val)], expected_seq_len=inp.layout.seq_len):
        patched = policy.forward(inp)["actions_norm"]
    np.testing.assert_array_equal(patched, base)


def test_embed_site_identity(policy, inp):
    base = policy.forward(inp)["actions_norm"]
    pos = inp.layout.group("agentview")
    with hooked(policy, captures=[Capture("embed", -1, pos)]) as s:
        policy.forward(inp)
    with hooked(policy, interventions=[Intervention("embed", -1, pos, "replace", s.get("embed", -1))]):
        patched = policy.forward(inp)["actions_norm"]
    np.testing.assert_array_equal(patched, base)


def test_nonidentity_patch_changes_output(policy, inp):
    base = policy.forward(inp)["actions_norm"]
    pos = inp.layout.group("readout")
    with hooked(policy, interventions=[Intervention("resid", 30, pos, "scale", 0.0)]):
        patched = policy.forward(inp)["actions_norm"]
    assert np.abs(patched - base).max() > 1e-3


def test_hooks_removed_after_exception(policy, inp):
    base = policy.forward(inp)["actions_norm"]
    with pytest.raises(ZeroDivisionError):
        with hooked(policy, interventions=[Intervention("resid", 5, [1, 2], "scale", 0.0)]):
            1 / 0
    assert all(len(l._forward_hooks) == 0 for l in policy.decoder_layers)
    np.testing.assert_array_equal(policy.forward(inp)["actions_norm"], base)


def test_missing_hook_fails(policy, inp):
    with pytest.raises(RuntimeError, match="unexpected number"):
        with hooked(policy, captures=[Capture("resid", 3)], n_forwards=2):
            policy.forward(inp)  # only one forward


def test_seq_len_mismatch_fails(policy, inp):
    with pytest.raises(RuntimeError, match="sequence length"):
        with hooked(policy, captures=[Capture("resid", 3)], expected_seq_len=inp.layout.seq_len + 1):
            policy.forward(inp)
