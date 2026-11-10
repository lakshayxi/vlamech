#!/usr/bin/env bash
# Mitigation arm selection (exploratory), mitigation held-out test and failure-monitor held-out test: closed-loop
# runs through one served model, then the analyses. Conditions and arms come from configs/protocol.yaml; the
# selected arms are frozen and committed before any held-out run. Resumable: finished episodes are skipped.
#
# Usage (after `source env/env.sh`, from a clean checkout):  bash scripts/run/monitor_and_mitigation.sh [shards=8]
# Log: results/logs/monitor_and_mitigation.log
set -uo pipefail
cd "$(dirname "$0")/../.."
N=${1:-8}
L=results/logs/monitor_and_mitigation.log
mkdir -p results/logs results/.policy
log() { echo "[chain] $(date -Is) $*" | tee -a "$L"; }
stop_server() { tmux kill-session -t vlamech-serve 2>/dev/null; }

dirty=$(git status --porcelain -- . ':(exclude)results')
[ -z "$dirty" ] || { log "working tree not clean: stopping"; echo "$dirty" >> "$L"; exit 1; }
log "commit $(git rev-parse --short HEAD)"

tmux has-session -t vlamech-serve 2>/dev/null || tmux new-session -d -s vlamech-serve \
  "cd $(pwd) && source env/env.sh && python scripts/run/serve_policy.py 2>&1 | tee results/.policy/serve.log"
for _ in $(seq 1 120); do [ -S results/.policy/oft.sock ] && break; sleep 5; done
[ -S results/.policy/oft.sock ] || { log "policy server did not start"; exit 1; }
trap stop_server EXIT

conditions() {  # conditions <protocol key>: one line per condition, "<name> <run_closed_loop args>"
  python - "$1" <<'PY'
import sys
sys.path.insert(0, "src")
from vlamech.analysis.arms import spec_of
from vlamech.config import protocol
for label in protocol(sys.argv[1]):
    s = spec_of(label)
    a = f"--family {s.family}" + (f" --value {s.value:g}" if s.value is not None else "") + f" --cameras {','.join(s.cameras)}"
    print(label.replace("@", "_"), a)
PY
}

run_list() {  # run_list <experiment> <states> <arm> <protocol key>
  local experiment=$1 states=$2 arm=$3 key=$4 name args
  while read -r name args; do
    log "$experiment $name $arm"
    # shellcheck disable=SC2086
    bash scripts/run/served.sh "$N" "${experiment}-${name}-${arm//:/-}" --experiment "$experiment" --states "$states" \
      --mitigation "$arm" $args >> "$L" 2>&1 || { log "$experiment $name $arm failed: stopping"; exit 1; }
  done < <(conditions "$key")
}

for arm in $(python -c "import sys; sys.path.insert(0,'src'); from vlamech.config import protocol; print(' '.join(protocol('mitigation.arms')))"); do
  run_list mitigation_selection exploratory "$arm" mitigation.selection_conditions
done
run_list mitigation_selection exploratory cag:1 mitigation.monitor_extra_fit_conditions

python scripts/analysis/mitigation.py --select >> "$L" 2>&1 || { log "selection failed"; exit 1; }
log "selection: $(grep -m1 Selected results/mitigation_selection/summary.md)"
git add configs/frozen_mitigation.yaml && git commit -q -m "freeze mitigation arms" \
  || { log "could not commit the frozen selection"; exit 1; }

for arm in $(python -c "import yaml; print(' '.join(yaml.safe_load(open('configs/frozen_mitigation.yaml'))['arms']))"); do
  run_list mitigation_test confirmatory "$arm" mitigation.selection_conditions
done
run_list failure_monitor confirmatory cag:1 mitigation.monitor_test_conditions
stop_server

python scripts/analysis/mitigation.py --confirm >> "$L" 2>&1 && log "mitigation analysed" || log "mitigation analysis failed"
python scripts/analysis/failure_monitor.py >> "$L" 2>&1 && log "monitor analysed" || log "monitor analysis failed"
log "done"
