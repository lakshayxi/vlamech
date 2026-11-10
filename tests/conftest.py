"""Shared fixtures. GPU/simulator tests are marked and skipped automatically when unavailable.

Run everything:      pytest -q
Only fast CPU tests: pytest -q -m "not gpu and not sim"
Verification run:    VLAMECH_REQUIRE_FULL=1 pytest -q   (a missing GPU / simulator is an ERROR, not a skip,
                     so a broken import can never pass as "skipped")
"""
import os

import pytest

GOAL_REPO = "moojink/openvla-7b-oft-finetuned-libero-goal"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs CUDA and the OFT checkpoint")
    config.addinivalue_line("markers", "sim: needs MuJoCo/LIBERO rendering")


REQUIRE_FULL = os.environ.get("VLAMECH_REQUIRE_FULL") == "1"


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available(), "CUDA not available"
    except Exception as e:  # reported in the skip reason, never swallowed silently
        return False, f"torch import failed: {e!r}"


def _has_libero():
    try:
        import libero.libero  # noqa: F401
        import mujoco  # noqa: F401

        return True, ""
    except Exception as e:
        return False, f"LIBERO/MuJoCo import failed: {e!r}"


def pytest_collection_modifyitems(config, items):
    (cuda_ok, cuda_why), (sim_ok, sim_why) = _has_cuda(), _has_libero()
    if REQUIRE_FULL and not (cuda_ok and sim_ok):
        raise pytest.UsageError(f"VLAMECH_REQUIRE_FULL=1 but: {'' if cuda_ok else cuda_why} {'' if sim_ok else sim_why}")
    for item in items:
        if "gpu" in item.keywords and not cuda_ok:
            item.add_marker(pytest.mark.skip(reason=cuda_why))
        if "sim" in item.keywords and not sim_ok:
            item.add_marker(pytest.mark.skip(reason=sim_why))


@pytest.fixture(scope="session")
def policy():
    from vlamech.config import model_revision
    from vlamech.models.oft import OFTPolicy
    from vlamech.utils.gpu import require_gpu

    require_gpu(min_free_gib=20)  # 7.5B params in bf16 ~ 15 GiB + activations; never displaces other users
    return OFTPolicy(GOAL_REPO, revision=model_revision(GOAL_REPO))


@pytest.fixture(scope="session")
def goal_env():
    from vlamech.envs.libero_env import LiberoTaskEnv

    env = LiberoTaskEnv("libero_goal", 0, segmentation=True)
    yield env
    env.close()


@pytest.fixture(scope="session")
def frame0(goal_env):
    return goal_env.reset(0)
