#!/usr/bin/env bash
# Held-out corruption sweep: every condition frozen in configs/frozen_levels.yaml, held-out init states, all 10
# tasks; the experiment requires videos, actions and captures. Conditions come from the freeze, never typed here.
#
# Usage (after `source env/env.sh`, one process per shard):
#   bash scripts/run_corruption_sweep.sh 0 2 &  bash scripts/run_corruption_sweep.sh 1 2
# Resumable (finished episodes are skipped). Logs: results/corruption_sweep/logs/<condition>.shard<i>.log
set -euo pipefail
cd "$(dirname "$0")/.."
SHARD=${1:?shard index}; N=${2:?number of shards}
LOGS=results/corruption_sweep/logs
CONDS=$(python -c "
import yaml
sel = yaml.safe_load(open('configs/frozen_levels.yaml'))
order = ['identity'] + sorted(f for f in sel['corruption'] if f != 'identity')
for f in order:
    for v in sel['corruption'][f]['levels']:
        print(f'{f}:{\"\" if v is None else f\"{v:g}\"}:')
for f in sel.get('variants', {}).get('exact_model_input', []):
    for v in sel['corruption'][f]['levels']:
        print(f'{f}:{v:g}:exactmi')
")
for c in $CONDS; do
  IFS=: read -r fam val var <<< "$c"
  mkdir -p "$LOGS"
  args=(--experiment corruption_sweep --states confirmatory --family "$fam" --shard "$SHARD/$N")
  [ -n "$val" ] && args+=(--value "$val")
  [ "$var" = "exactmi" ] && args+=(--exact-model-input)
  tag="$fam${val:+-$val}${var:+-$var}"
  echo "[sweep] $(date -Is) $tag shard $SHARD/$N"
  python scripts/run_closed_loop.py "${args[@]}" > "$LOGS/$tag.shard$SHARD.log" 2>&1
  grep -E "^\[run\] .*shard" "$LOGS/$tag.shard$SHARD.log" | head -1 || true
done
echo "[sweep] $(date -Is) done"
