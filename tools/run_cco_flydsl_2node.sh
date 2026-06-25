#!/bin/bash
# Launch a 2-rank FlyDSL+cco example across two nodes (rank 0 local, rank 1 remote).
#
# Bootstraps without cross-host MPI: rank 0's host generates the cco UniqueId
# (embedding its RDMA-iface address), rsyncs it to the remote node, then both
# ranks read it via the example's env/file bootstrap (CCO_RANK/CCO_WORLD/CCO_UID_FILE).
#
# Prereqs (both nodes, in $CONTAINER): flydsl installed, the cco host ext built,
# and $MORI/{python, lib/libmori_cco_device.bc, build_ccomerge/*.so} present
# (rsync the local tree to the remote node at the same path first).
#
# Usage:
#   bash tools/run_cco_flydsl_2node.sh [example_rel_path]
# Env overrides: REMOTE_HOST CONTAINER IFACE MORI_CCO_GDA_CONN
set -uo pipefail

MORI=/home/jiahzhou/workspace/mori
REMOTE="${REMOTE_HOST:-a07u25}"
CONTAINER="${CONTAINER:-mori_cco_test}"
IFACE="${IFACE:-enp159s0np0}"
CONN="${MORI_CCO_GDA_CONN:-crossnode}"
EXAMPLE="${1:-examples/cco/03_flydsl_put/main.py}"
UID_FILE="$MORI/cco_flydsl_uid_$$.bin"
LOG="/tmp/cco_flydsl_2node_$$"; mkdir -p "$LOG"

LLP="$MORI/build_ccomerge/src/application:$MORI/build_ccomerge/src/cco:$MORI/build_ccomerge/src/collective:$MORI/build_ccomerge/src/io:$MORI/build_ccomerge/src/metrics:$MORI/build_ccomerge/src/ops:$MORI/build_ccomerge/src/shmem"

ENVSH="cd $MORI; \
export LD_LIBRARY_PATH=$LLP:\$LD_LIBRARY_PATH; \
export PYTHONPATH=$MORI/python; \
export MORI_CCO_BC=$MORI/lib/libmori_cco_device.bc; \
export MORI_SOCKET_IFNAME=$IFACE; \
export MORI_CCO_GDA_CONN=$CONN; \
export CCO_WORLD=2; export CCO_GPU=0; export CCO_UID_FILE=$UID_FILE"

echo "[launch] example=$EXAMPLE conn=$CONN iface=$IFACE  rank0=$(hostname) rank1=$REMOTE"

cleanup() { rm -f "$UID_FILE"; ssh "$REMOTE" "rm -f $UID_FILE" 2>/dev/null || true; }
trap cleanup EXIT

# 1. Generate the UniqueId on rank 0's host (embeds this host's RDMA address).
rm -f "$UID_FILE"
docker exec "$CONTAINER" bash -lc "$ENVSH; python3 -c 'import os; from mori.cco import Communicator; open(os.environ[\"CCO_UID_FILE\"],\"wb\").write(bytes(Communicator.get_unique_id()))'"
[ -s "$UID_FILE" ] || { echo "ERROR: uid not generated"; exit 1; }

# 2. Push the UniqueId to the remote node.
rsync -a "$UID_FILE" "$REMOTE:$UID_FILE"

# 3. Remote rank 1 (background).
ssh "$REMOTE" "sudo -n docker exec $CONTAINER bash -lc '$ENVSH; export CCO_RANK=1; python3 $EXAMPLE'" > "$LOG/rank1.log" 2>&1 &
RPID=$!

# 4. Local rank 0 (foreground).
docker exec "$CONTAINER" bash -lc "$ENVSH; export CCO_RANK=0; python3 $EXAMPLE" > "$LOG/rank0.log" 2>&1
wait $RPID 2>/dev/null

echo "==================== rank 0 ($(hostname)) ===================="; cat "$LOG/rank0.log"
echo "==================== rank 1 ($REMOTE) ===================="; cat "$LOG/rank1.log"
