#!/usr/bin/env bash
# Severity calibration: every ladder level of every implemented family on the calibration split
# (protocol corruption.selection_rule), with traces and videos. Levels come from configs/protocol.yaml.
#
# Usage (after `source env/env.sh`, one process per shard):
#   bash scripts/run/severity_calibration.sh 0 2 &  bash scripts/run/severity_calibration.sh 1 2
# All shards walk the levels in the same order, so they work on the same level at the same time. Re-running
# resumes (finished episodes are skipped by the runner). Logs: results/severity_calibration/logs/<family>-<value>.shard<i>.log
set -euo pipefail
cd "$(dirname "$0")/../.."
SHARD=${1:?shard index}; N=${2:?number of shards}
LOGS=results/severity_calibration/logs; mkdir -p "$LOGS"
LEVELS=$(python -c "
import sys; sys.path.insert(0, 'src')
from vlamech.perturbations.image import LADDER_KEY, ladder
for f in LADDER_KEY:
    for v in ladder(f):
        print(f'{f}:{v:g}')
")
for lv in $LEVELS; do
  fam=${lv%%:*}; val=${lv#*:}
  mkdir -p "$LOGS"
  echo "[sweep] $(date -Is) $fam $val shard $SHARD/$N"
  python scripts/run/closed_loop.py --experiment severity_calibration --states calibration --family "$fam" --value "$val" \
    --record-video --shard "$SHARD/$N" > "$LOGS/$fam-$val.shard$SHARD.log" 2>&1
  grep -E "^\[run\] .*shard" "$LOGS/$fam-$val.shard$SHARD.log" | head -1 || true
done
echo "[sweep] $(date -Is) done"
