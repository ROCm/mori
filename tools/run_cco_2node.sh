#!/bin/bash
# Cross-host launcher for tests/cpp/cco/test_cco_multiprocess.
#
# Spawns N ranks per host (default 8 = local GPU count) on this node and one
# remote node. Each rank runs inside the docker container as a separate
# process; bootstrap uses SocketBootstrapNetwork with a UniqueId generated on
# rank-0's host and rsync'd to the other.
#
# Defaults are wired for the current dev setup (a07u19 ↔ a07u25, container
# mori_cco_test, 10.245.128.x via enp159s0np0). Override via env vars.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-a07u25}"
CONTAINER="${CONTAINER:-mori_cco_test}"
IFACE="${IFACE:-enp159s0np0}"
PORT="${PORT:-18459}"
NRANKS_PER_HOST="${NRANKS_PER_HOST:-8}"
BINARY_IN_CONTAINER="${BINARY_IN_CONTAINER:-/workspace/mori/build_docker/tests/cpp/cco/test_cco_multiprocess}"
UID_IN_CONTAINER="${UID_IN_CONTAINER:-/workspace/mori/cco_uid_$$.bin}"
UID_ON_HOST="${UID_ON_HOST:-/home/jiahzhou/workspace/mori/cco_uid_$$.bin}"

WORLD=$(( NRANKS_PER_HOST * 2 ))
LOG_DIR="${LOG_DIR:-/tmp/cco_2node_$$}"
mkdir -p "$LOG_DIR"

echo "[launch] WORLD=$WORLD ranks 0..$((NRANKS_PER_HOST-1)) on $(hostname), $NRANKS_PER_HOST..$((WORLD-1)) on $REMOTE_HOST"
echo "[launch] iface=$IFACE port=$PORT uid=$UID_ON_HOST logs=$LOG_DIR"

# 1. Generate UID on this host (rank-0 side).
sudo -n docker exec "$CONTAINER" "$BINARY_IN_CONTAINER" \
  --gen-uid "$IFACE" "$PORT" "$UID_IN_CONTAINER"

# 2. Push UID file to remote.
rsync -a "$UID_ON_HOST" "$REMOTE_HOST:$UID_ON_HOST"

# Cleanup helper.
cleanup() {
  rm -f "$UID_ON_HOST"
  ssh "$REMOTE_HOST" "rm -f $UID_ON_HOST" 2>/dev/null || true
}
trap cleanup EXIT

# 3. Spawn remote ranks (background).
ssh "$REMOTE_HOST" "
  set -e
  mkdir -p $LOG_DIR
  declare -a PIDS=()
  for R in \$(seq $NRANKS_PER_HOST $((WORLD-1))); do
    sudo -n docker exec -e MORI_SOCKET_IFNAME=$IFACE $CONTAINER $BINARY_IN_CONTAINER \
      --rank \$R --world $WORLD --uid-file $UID_IN_CONTAINER \
      --gpu-offset $NRANKS_PER_HOST > $LOG_DIR/rank_\$R.log 2>&1 &
    PIDS+=(\$!)
  done
  RC=0
  for pid in \${PIDS[@]}; do wait \$pid || RC=\$((RC|\$?)); done
  exit \$RC
" > "$LOG_DIR/remote_orchestrator.log" 2>&1 &
SSH_PID=$!

# 4. Spawn local ranks.
declare -a PIDS=()
for R in $(seq 0 $((NRANKS_PER_HOST-1))); do
  sudo -n docker exec -e MORI_SOCKET_IFNAME="$IFACE" "$CONTAINER" "$BINARY_IN_CONTAINER" \
    --rank "$R" --world "$WORLD" --uid-file "$UID_IN_CONTAINER" \
    --gpu-offset 0 > "$LOG_DIR/rank_$R.log" 2>&1 &
  PIDS+=($!)
done

RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=$((RC | $?)); done

# 5. Reap remote.
wait "$SSH_PID" || RC=$((RC | $?))

echo
echo "================ rank logs ================"
for R in $(seq 0 $((NRANKS_PER_HOST-1))); do
  echo "--- rank $R (local: $(hostname)) ---"
  cat "$LOG_DIR/rank_$R.log"
done

ssh "$REMOTE_HOST" "for R in \$(seq $NRANKS_PER_HOST $((WORLD-1))); do
  echo '--- rank '\$R' (remote: $REMOTE_HOST) ---'
  cat $LOG_DIR/rank_\$R.log
done"

echo "================ exit code: $RC ================"
exit $RC
