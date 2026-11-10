#!/usr/bin/env bash
# Run one closed-loop condition as N shards that share one served model (scripts/serve_policy.py).
#
# Usage (after `source env/env.sh`, with the server running, e.g.
#   tmux new-session -d -s vlamech-serve "mkdir -p results/.policy && python scripts/serve_policy.py 2>&1 | tee results/.policy/serve.log"):
#   bash scripts/run_served.sh <n_shards> <name> <run_closed_loop args...>
# e.g.
#   bash scripts/run_served.sh 8 clean_capture --experiment clean_capture --states exploratory
#
# Starts tmux sessions vlamech-<name>-<i> (i = 0..n-1), each sourcing env/env.sh itself (a new tmux session does not
# inherit the caller's environment), then `run_closed_loop.py <args> --shard i/n --policy-server <socket>`, logs to
# results/<experiment>/logs/<name>.shard<i>.log, and returns when all have ended. Re-running resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:?number of shards}; NAME=${2:?run name for sessions and logs}; shift 2
SOCK=${POLICY_SOCKET:-results/.policy/oft.sock}
[ -S "$SOCK" ] || { echo "no policy server at $SOCK (start scripts/serve_policy.py first)"; exit 1; }
EXPERIMENT=$(python -c "import sys; a=sys.argv[1:]; print(a[a.index('--experiment')+1])" "$@")
LOGS=results/$EXPERIMENT/logs; mkdir -p "$LOGS"
NAME=${NAME//./_}  # tmux replaces '.' in session names
for ((i = 0; i < N; i++)); do
  s="vlamech-$NAME-$i"
  tmux has-session -t "$s" 2>/dev/null && { echo "session $s exists"; exit 1; }
  tmux new-session -d -s "$s" \
    "cd $(pwd) && source env/env.sh && python scripts/run_closed_loop.py $(printf '%q ' "$@") --shard $i/$N --policy-server $SOCK > $LOGS/$NAME.shard$i.log 2>&1"
done
echo "[served] $(date -Is) $N shards of $NAME started"
for ((i = 0; i < N; i++)); do
  while tmux has-session -t "vlamech-$NAME-$i" 2>/dev/null; do sleep 20; done
done
echo "[served] $(date -Is) $NAME done"
grep -h "^\[run\] .*shard" "$LOGS"/$NAME.shard*.log | head -"$N" || true
