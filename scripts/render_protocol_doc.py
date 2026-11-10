"""Render configs/protocol.yaml into docs/hyperparameters.md (one table row per parameter).

Usage:  python scripts/render_protocol_doc.py           # write the doc
        python scripts/render_protocol_doc.py --check   # exit 1 if the committed doc is stale (used by tests)

Why generated: the doc must never disagree with the values the code actually uses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.config import _is_param, protocol_tree  # noqa: E402

DOC = ROOT / "docs" / "hyperparameters.md"
HEADER = """# Hyperparameters and design parameters

*Generated from `configs/protocol.yaml` by `scripts/render_protocol_doc.py`. Do not edit by hand.*

Every number used by the experiments is listed here with its category and justification.

| Category | Meaning |
|---|---|
| `upstream` | fixed by a cited prior protocol (OpenVLA-OFT / LIBERO evaluation, or a replicated paper); changing it breaks comparability |
| `definitional` | follows from the definition of the measured quantity; no free choice |
| `rule` | the *value* is the output of a pre-registered selection rule applied to calibration/exploratory data only; the rule (or the ladder it searches) is what is fixed here; selected values are frozen in `configs/frozen_levels.yaml` |
| `convention` | standard statistical or methodological convention (cited) |

"""


def _fmt(v) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    if isinstance(v, list):
        return "[" + ", ".join(map(str, v)) + "]"
    s = str(v).strip().replace("\n", " ")
    return s.replace("|", "\\|")


def render() -> str:
    tree = protocol_tree()
    out = [HEADER]
    for section, node in tree.items():
        out.append(f"## `{section}`\n\n| Parameter | Value | Category | Justification |\n|---|---|---|---|\n")
        rows = []

        def walk(n, path):
            if _is_param(n):
                rows.append(f"| `{path}` | {_fmt(n['value'])} | {n['source']} | {_fmt(n['why'])} |")
                return
            for k, v in n.items():
                walk(v, f"{path}.{k}")

        walk(node, section)
        out.append("\n".join(rows) + "\n\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render()
    if args.check:
        ok = DOC.exists() and DOC.read_text() == text
        print("up to date" if ok else "docs/hyperparameters.md is stale: run scripts/render_protocol_doc.py")
        sys.exit(0 if ok else 1)
    DOC.write_text(text)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
