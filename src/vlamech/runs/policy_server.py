"""One model copy on the GPU serving many episode processes (throughput; frees GPU memory for other work).

Used by: scripts/serve_policy.py (server), scripts/run_closed_loop.py --policy-server (clients), tests/test_policy_server.py.

Why: one closed-loop process holds a full 7.5B model (~19 GB) and spends most of an episode in the simulator and
renderer, so the GPU idles. Here the model is loaded once; any number of runner processes (each with its own
simulator, on its own CPU core) send their queries to it over a local socket.

Exactness: the server answers requests ONE AT A TIME with exactly the calls the runner would make in-process
(policy.prepare + policy.forward, policy.act_many), on the same inputs, with batch size 1. Nothing is batched, so an
action computed through the server is the action the in-process policy computes (tested against in-process calls,
tests/test_policy_server.py; GPU parity in the same file). The client checks at connect time that the server holds
the model the run's manifest names (repo, revision, settings) and refuses otherwise.

Wire format: multiprocessing.connection over a Unix socket, authenticated with a per-server random key stored next
to the socket (owner-only file). Requests: ("info",), ("forward", request dict), ("act_many", args), ("close",).
"""
from __future__ import annotations

import os
import secrets
import threading
from multiprocessing.connection import Client, Listener, wait
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROTOCOL = 1


def _key_path(address: Path) -> Path:
    return Path(str(address) + ".key")


class _Request(dict):
    """What OFTPolicy.prepare would have received; built client-side, prepared server-side."""


def _arrays(images_224) -> Optional[list]:
    return None if images_224 is None else [np.asarray(im, dtype=np.uint8) for im in images_224]


def _pil(arrays) -> Optional[tuple]:
    if arrays is None:
        return None
    from PIL import Image

    return tuple(Image.fromarray(a) for a in arrays)


def handle(policy, msg):
    """Answer one request with the same calls the runner makes in-process."""
    kind = msg[0]
    if kind == "forward":
        r = msg[1]
        out = policy.forward(policy.prepare(r["agentview"], r["wrist"], r["proprio"], r["instruction"],
                                            images_224=_pil(r["images_224"])))
        return {"actions": out["actions"], "actions_norm": out["actions_norm"]}
    if kind == "act_many":
        agentview, wrist, proprio, instructions = msg[1]
        out = policy.act_many(agentview, wrist, proprio, list(instructions))
        return {ins: {"actions": o["actions"], "actions_norm": o["actions_norm"]} for ins, o in out.items()}
    if kind == "unnormalize":
        return policy.unnormalize(msg[1])
    raise ValueError(f"unknown request {kind!r}")


def serve(address: Path, policy, info: dict, stop: Optional[threading.Event] = None, log=print) -> None:
    """Serve `policy` at the Unix socket `address` until `stop` is set (or forever). Requests are answered in arrival
    order, one at a time."""
    address = Path(address)
    if address.exists():
        raise FileExistsError(f"{address} exists: another server, or a stale socket (remove it if no server runs)")
    key = secrets.token_bytes(32)
    kp = _key_path(address)
    fd = os.open(kp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    listener = Listener(str(address), family="AF_UNIX", authkey=key)
    conns: list = []
    lock = threading.Lock()
    stop = stop or threading.Event()

    def accept():
        while not stop.is_set():
            try:
                c = listener.accept()
            except OSError:
                return
            with lock:
                conns.append(c)

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    full_info = {"protocol": PROTOCOL, **info}
    n = 0
    try:
        while not stop.is_set():
            with lock:
                current = list(conns)
            for c in wait(current, timeout=0.2) if current else []:
                try:
                    msg = c.recv()
                except (EOFError, OSError):
                    with lock:
                        conns.remove(c)
                    continue
                if msg[0] == "info":
                    c.send(full_info)
                    continue
                if msg[0] == "close":
                    c.close()
                    with lock:
                        conns.remove(c)
                    continue
                try:
                    c.send(("ok", handle(policy, msg)))
                except Exception as e:  # reported to the client, which raises (never a silent wrong action)
                    c.send(("error", f"{type(e).__name__}: {e}"))
                n += 1
                if n % 1000 == 0:
                    log(f"[server] {n} requests, {len(current)} clients", flush=True)
            if not current:
                stop.wait(0.05)
    finally:
        stop.set()
        listener.close()
        for p in (address, kp):
            p.unlink(missing_ok=True)


class RemotePolicy:
    """Client side: the subset of OFTPolicy that run_episode and the mitigations use."""

    def __init__(self, address: Path, expect: dict):
        address = Path(address)
        self.conn = Client(str(address), family="AF_UNIX", authkey=_key_path(address).read_bytes())
        self.conn.send(("info",))
        self.info = self.conn.recv()
        want = {"protocol": PROTOCOL, **expect}
        bad = {k: (self.info.get(k), v) for k, v in want.items() if self.info.get(k) != v}
        if bad:
            self.close()
            raise RuntimeError(f"policy server does not hold the model this run needs: {bad}")
        self.center_crop = self.info["center_crop"]

    def _call(self, msg):
        self.conn.send(msg)
        status, val = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"policy server: {val}")
        return val

    def prepare(self, agentview, wrist, proprio, instruction: str, images_224=None) -> _Request:
        return _Request(agentview=np.asarray(agentview), wrist=np.asarray(wrist), proprio=np.asarray(proprio),
                        instruction=instruction, images_224=_arrays(images_224))

    def forward(self, req: _Request) -> dict:
        return self._call(("forward", dict(req)))

    def act_many(self, agentview, wrist, proprio, instructions: Sequence[str]) -> dict:
        return self._call(("act_many", (np.asarray(agentview), np.asarray(wrist), np.asarray(proprio), list(instructions))))

    def unnormalize(self, norm):
        return self._call(("unnormalize", np.asarray(norm)))

    def close(self):
        try:
            self.conn.send(("close",))
        except OSError:
            pass
        self.conn.close()
