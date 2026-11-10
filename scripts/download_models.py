"""Download the model checkpoints into the Hugging Face cache and pin them.

Usage:  python scripts/download_models.py [repo_id ...] [--repin]   (default: the OFT checkpoint and pi0.5)

configs/models.lock.yaml holds, for each repo, the commit and the SHA-256 of every file. For a pinned repo this script
downloads that commit and verifies the files against the lock; --repin moves the pin (a protocol change). Experiments
load models only at the pinned commit and run with HF_HUB_OFFLINE=1 (env/env.sh), so nothing is fetched during a run.
pi0.5 also needs the tokenizer of google/paligemma-3b-pt-224, a gated repo: accept its license on the Hub and log in
(`huggingface-cli login`) before running this script.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlamech.config import protocol  # noqa: E402
from vlamech.utils.io import atomic_write_bytes, file_sha256  # noqa: E402

LOCK = ROOT / "configs" / "models.lock.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*")
    ap.add_argument("--repin", action="store_true")
    args = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "0"  # this is the one place that downloads
    from huggingface_hub import snapshot_download

    from vlamech.models import pi05

    def patterns(repo):
        if repo == pi05.REPO_ID:  # never import the OFT module in the LeRobot environment
            return pi05.DOWNLOAD_ALLOW, None
        from vlamech.models.oft import DOWNLOAD_ALLOW, DOWNLOAD_IGNORE

        return DOWNLOAD_ALLOW, DOWNLOAD_IGNORE

    repos = args.repos or [protocol("policy.checkpoint"), protocol("policies.pi05.checkpoint")]
    lock = yaml.safe_load(LOCK.read_text()) if LOCK.exists() else {}
    lock = lock or {}
    for repo in repos:
        pinned = lock.get(repo, {}).get("revision")
        rev = None if (pinned is None or args.repin) else pinned
        allow, ignore = patterns(repo)
        path = Path(snapshot_download(repo, revision=rev, allow_patterns=allow, ignore_patterns=ignore))
        files = {str(f.relative_to(path)): file_sha256(f) for f in sorted(path.rglob("*")) if f.is_file()}
        if rev is not None:
            bad = {k for k, v in lock[repo]["files"].items() if files.get(k) != v}
            if bad:
                sys.exit(f"{repo}: files differ from the lock: {sorted(bad)}")
            print(f"[ok] {repo} @ {rev}: {len(files)} files match the lock")
            continue
        lock[repo] = {"revision": path.name, "files": files}
        print(f"[pinned] {repo} @ {path.name}: {len(files)} files")
    if pi05.REPO_ID in repos:  # tokenizer only; not part of the model lock
        snapshot_download(pi05.TOKENIZER_REPO, allow_patterns=["tokenizer*", "special_tokens_map.json", "added_tokens.json"])
    atomic_write_bytes(LOCK, ("# Written by scripts/download_models.py. Do not edit by hand.\n"
                              + yaml.safe_dump(lock, sort_keys=True)).encode())


if __name__ == "__main__":
    main()
