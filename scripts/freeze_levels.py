"""Freeze the calibrated corruption levels and the oracle thresholds into configs/frozen_levels.yaml.

Usage:  python scripts/freeze_levels.py   (refuses to overwrite an existing freeze)

Inputs (hash-recorded): results/severity_calibration/summary.json (levels selected by the protocol rule) and
results/oracle_calibration/thresholds.json. Frozen conditions: identity (the paired clean reference), every family's
selected levels, the two controls (target_occlusion, matched_bg_box), and the exact-model-input variant of the
background-only family. Once this file exists, held-out runs accept nothing else (vlamech.runs.experiments.preflight).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from vlamech.config import FROZEN_LEVELS  # noqa: E402
from vlamech.utils.io import file_sha256, read_json  # noqa: E402

CALIBRATION = ROOT / "results" / "severity_calibration" / "summary.json"
ORACLE = ROOT / "results" / "oracle_calibration" / "thresholds.json"


def main():
    if FROZEN_LEVELS.exists():
        raise SystemExit(f"{FROZEN_LEVELS} exists: a freeze is never overwritten")
    cal, oracle = read_json(CALIBRATION), read_json(ORACLE)
    pending = {f: v["extension"] for f, v in cal["families"].items() if isinstance(v.get("extension"), str)
               and "not yet run" in v["extension"]}
    if pending:
        raise SystemExit(f"ladder extensions still due: {pending}")
    corruption = {"identity": {"levels": [None]}, "target_occlusion": {"levels": [None]},
                  "matched_bg_box": {"levels": [None]}}
    for fam, v in cal["families"].items():
        corruption[fam] = {"levels": [float(x) for x in v["selected_levels"]],
                           "unreachable_targets": v["unreachable_targets"], "benign": v["benign"]}
    sel = {
        "frozen": dt.datetime.now().isoformat(timespec="seconds"),
        "corruption": corruption,
        "variants": {"exact_model_input": ["bg_patches"]},
        "oracle": {"classifier_version": oracle["classifier_version"], "trace_schema": oracle["trace_schema"],
                   "thresholds_file": str(ORACLE.relative_to(ROOT)), "thresholds_sha256": file_sha256(ORACLE)},
        "sources": {"calibration_summary_sha256": file_sha256(CALIBRATION)},
    }
    FROZEN_LEVELS.write_text("# Corruption levels and oracle thresholds frozen after calibration, before any held-out run.\n"
                             "# Written once by scripts/freeze_levels.py, never edited.\n" + yaml.safe_dump(sel, sort_keys=False))
    print(FROZEN_LEVELS.read_text())


if __name__ == "__main__":
    main()
