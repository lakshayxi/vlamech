"""Serve the protocol's policy from one GPU process to many runner processes (vlamech.runs.policy_server).

Usage (after `source env/env.sh`):
  mkdir -p results/.policy && python scripts/run/serve_policy.py 2>&1 | tee results/.policy/serve.log
  python scripts/run/closed_loop.py ... --shard 0/8 --policy-server results/.policy/oft.sock   (x 8 shards)

Loads exactly what an in-process run loads (protocol policy.checkpoint at the pinned revision, policy.center_crop,
policy.global_seed) and answers one request at a time, so served actions equal in-process actions
(tests/test_policy_server.py). Clients refuse a server holding a different model or settings. Ctrl-C stops it and
removes the socket.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.config import model_revision, protocol  # noqa: E402

DEFAULT_ADDRESS = ROOT / "results" / ".policy" / "oft.sock"
GPU_GIB = 20  # same reservation as one in-process runner (scripts/run/closed_loop.py)


def expected_info(settings_get=protocol) -> dict:
    """What a client checks the server against (built from the protocol, as the runner builds its manifest)."""
    repo = settings_get("policy.checkpoint")
    return {"repo": repo, "revision": model_revision(repo), "center_crop": bool(settings_get("policy.center_crop")),
            "global_seed": int(settings_get("policy.global_seed"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", type=Path, default=DEFAULT_ADDRESS)
    a = ap.parse_args()
    from vlamech.models.oft import OFTPolicy
    from vlamech.runs.policy_server import serve
    from vlamech.utils.gpu import require_gpu
    from vlamech.utils.seeding import set_global_seed

    info = expected_info()
    require_gpu(GPU_GIB)
    set_global_seed(info["global_seed"])
    policy = OFTPolicy(info["repo"], revision=info["revision"], center_crop=info["center_crop"])
    a.address.parent.mkdir(parents=True, exist_ok=True)
    print(f"[server] {info['repo']}@{info['revision'][:12]} serving at {a.address}", flush=True)
    try:
        serve(a.address, policy, info)
    except KeyboardInterrupt:
        print("[server] stopped", flush=True)


if __name__ == "__main__":
    main()
