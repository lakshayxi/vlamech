"""Policy server: answers equal in-process calls; wrong model refused; errors surface. CPU (fake policy) + GPU parity."""
import threading
import time

import numpy as np
import pytest

from vlamech.runs.policy_server import RemotePolicy, serve


class FakePolicy:
    """Deterministic stand-in with OFTPolicy's call surface: actions depend on every input."""
    center_crop = True

    def prepare(self, agentview, wrist, proprio, instruction, images_224=None):
        extra = 0.0 if images_224 is None else float(np.asarray(images_224[0]).mean())
        return (float(np.asarray(agentview).mean() - np.asarray(wrist).mean()) + float(np.sum(proprio)) + extra,
                instruction)

    def forward(self, inp):
        x, ins = inp
        if ins == "boom":
            raise ValueError("bad input")
        norm = np.full((8, 7), x + len(ins), np.float32)
        return {"actions_norm": norm, "actions": self.unnormalize(norm), "hidden_states": "not sent"}

    def unnormalize(self, norm):
        return np.asarray(norm) * 2.0 + 1.0

    def act_many(self, agentview, wrist, proprio, instructions):
        return {i: self.forward(self.prepare(agentview, wrist, proprio, i)) for i in instructions}


INFO = {"repo": "fake/repo", "revision": "abc", "center_crop": True}


@pytest.fixture
def server(tmp_path):
    addr = tmp_path / "policy.sock"
    stop = threading.Event()
    t = threading.Thread(target=serve, args=(addr, FakePolicy(), INFO, stop, lambda *a, **k: None), daemon=True)
    t.start()
    for _ in range(100):
        if addr.exists() and (tmp_path / "policy.sock.key").exists():
            break
        time.sleep(0.02)
    yield addr
    stop.set()
    t.join(5)


def test_remote_equals_in_process(server):
    rng = np.random.default_rng(0)
    av, wr = rng.integers(0, 255, (256, 256, 3), np.uint8), rng.integers(0, 255, (256, 256, 3), np.uint8)
    pr = rng.normal(size=8)
    local, remote = FakePolicy(), RemotePolicy(server, INFO)
    try:
        a = local.forward(local.prepare(av, wr, pr, "put the bowl on the stove"))
        b = remote.forward(remote.prepare(av, wr, pr, "put the bowl on the stove"))
        np.testing.assert_array_equal(a["actions"], b["actions"])
        np.testing.assert_array_equal(a["actions_norm"], b["actions_norm"])
        assert "hidden_states" not in b
        views = (av[:224, :224], wr[:224, :224])
        from PIL import Image
        pil = tuple(Image.fromarray(v) for v in views)
        np.testing.assert_array_equal(local.forward(local.prepare(av, wr, pr, "x", images_224=pil))["actions"],
                                      remote.forward(remote.prepare(av, wr, pr, "x", images_224=pil))["actions"])
        m1, m2 = local.act_many(av, wr, pr, ["a", ""]), remote.act_many(av, wr, pr, ["a", ""])
        for k in m1:
            np.testing.assert_array_equal(m1[k]["actions_norm"], m2[k]["actions_norm"])
        np.testing.assert_array_equal(remote.unnormalize(np.ones((8, 7))), local.unnormalize(np.ones((8, 7))))
        assert remote.center_crop is True
    finally:
        remote.close()


def test_two_clients_and_errors_surface(server):
    a, b = RemotePolicy(server, INFO), RemotePolicy(server, INFO)
    try:
        img = np.zeros((4, 4, 3), np.uint8)
        assert a.forward(a.prepare(img, img, np.zeros(8), "ab"))["actions_norm"][0, 0] == 2
        assert b.forward(b.prepare(img, img, np.ones(8), "ab"))["actions_norm"][0, 0] == 10
        with pytest.raises(RuntimeError, match="bad input"):
            a.forward(a.prepare(img, img, np.zeros(8), "boom"))
        assert a.forward(a.prepare(img, img, np.zeros(8), "ab"))["actions_norm"][0, 0] == 2  # still usable
    finally:
        a.close(), b.close()


def test_wrong_model_is_refused(server):
    with pytest.raises(RuntimeError, match="does not hold the model"):
        RemotePolicy(server, INFO | {"revision": "other"})


def test_stale_socket_is_not_overwritten(tmp_path):
    (tmp_path / "s.sock").write_text("")
    with pytest.raises(FileExistsError):
        serve(tmp_path / "s.sock", FakePolicy(), INFO)


@pytest.mark.gpu
def test_gpu_server_matches_in_process_oft(tmp_path, policy, frame0):
    """The served OFT policy returns bit-identical actions to in-process calls on the same inputs."""
    info = {"repo": policy.repo_id, "revision": policy.revision, "center_crop": policy.center_crop}
    addr, stop = tmp_path / "p.sock", threading.Event()
    t = threading.Thread(target=serve, args=(addr, policy, info, stop, lambda *a, **k: None), daemon=True)
    t.start()
    while not ((tmp_path / "p.sock.key").exists() and (tmp_path / "p.sock").exists()):
        time.sleep(0.05)
    remote = RemotePolicy(addr, info)
    try:
        f, ins = frame0, "put the bowl on the stove"
        a = policy.forward(policy.prepare(f.agentview, f.wrist, f.proprio, ins))
        b = remote.forward(remote.prepare(f.agentview, f.wrist, f.proprio, ins))
        np.testing.assert_array_equal(a["actions_norm"], b["actions_norm"])
        np.testing.assert_array_equal(a["actions"], b["actions"])
        m1 = policy.act_many(f.agentview, f.wrist, f.proprio, [ins, ""])
        m2 = remote.act_many(f.agentview, f.wrist, f.proprio, [ins, ""])
        for k in m1:
            np.testing.assert_array_equal(m1[k]["actions_norm"], m2[k]["actions_norm"])
    finally:
        remote.close()
        stop.set()
        t.join(10)


class Recording(FakePolicy):
    """Server-side: keeps every request it received and the answer it gave."""

    def __init__(self):
        self.log = []

    def prepare(self, agentview, wrist, proprio, instruction, images_224=None):
        self.log.append({"agentview": np.array(agentview), "wrist": np.array(wrist), "proprio": np.array(proprio)})
        return super().prepare(agentview, wrist, proprio, instruction, images_224)

    def forward(self, inp):
        out = super().forward(inp)
        self.log[-1]["actions_norm"] = out["actions_norm"]
        return out


@pytest.mark.sim
def test_episode_through_server_is_exact(goal_env, tmp_path):
    """A real (corrupted) episode through the server: the server receives exactly the images the episode fed it
    (captured), and the episode executes exactly the actions the server computed. Independent of renderer
    nondeterminism (two runs of the same episode may differ by a pixel)."""
    from vlamech.envs.episode import run_episode
    from vlamech.perturbations.image import CorruptionSpec, EpisodeCorruptor

    served = Recording()
    addr, stop = tmp_path / "e.sock", threading.Event()
    t = threading.Thread(target=serve, args=(addr, served, INFO, stop, lambda *a, **k: None), daemon=True)
    t.start()
    while not ((tmp_path / "e.sock.key").exists() and (tmp_path / "e.sock").exists()):
        time.sleep(0.02)
    remote = RemotePolicy(addr, INFO)
    try:
        res, extra = run_episode(goal_env, remote, 0, EpisodeCorruptor(CorruptionSpec("noise", 32.0), "task0-init0"),
                                 max_steps=40, chunk=8, record_actions=True, capture=True)
    finally:
        remote.close(), stop.set(), t.join(5)
    assert res.error is None, res.error
    cap = extra["capture"]
    assert len(served.log) == res.n_queries == len(cap["step"]) == len(extra["actions_norm"])
    for k, req in enumerate(served.log):
        np.testing.assert_array_equal(req["agentview"], cap["fed_agentview"][k])
        np.testing.assert_array_equal(req["wrist"], cap["fed_wrist"][k])
        np.testing.assert_array_equal(req["proprio"], cap["proprio"][k])
        np.testing.assert_array_equal(req["actions_norm"], extra["actions_norm"][k])
