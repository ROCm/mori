#!/bin/bash
# Two-node check that a master-side heartbeat-interval change reaches an
# already-registered peer over a real network, with neither process restarted.
#
# Node 0 runs the master (and the admin CLI + metrics scrape, both local to it).
# Node 1 runs a peer.  The peer's heartbeat count is read out of the master's
# Prometheus endpoint, so the cadence is measured where it actually lands
# rather than inferred from the peer's own logs.
#
# Submit:  sbatch tests/slurm/hb_interval_runtime.sbatch
#SBATCH --job-name=umbp_hb_runtime
#SBATCH --partition=Compute-Group01
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/apps/yutongwu/store/mori/tests/slurm/hb_runtime_%j.log

set -u

REPO=/apps/yutongwu/store/mori
B=$REPO/build
DEPS=$REPO/slurm_deps
export LD_LIBRARY_PATH=$DEPS:$B/src/application:$B/src/cco:$B/src/collective:$B/src/io:$B/src/metrics:$B/src/ops:$B/src/shmem:${LD_LIBRARY_PATH:-}

# Distinct high ports so a co-scheduled job on the same node cannot collide.
GRPC_PORT=51077
METRICS_PORT=19077

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_NODE=${NODES[0]}
PEER_NODE=${NODES[1]}
# The batch script itself runs on node 0, so the master is local to it and the
# admin CLI / curl calls below need no extra hop.
#
# Address selection is the whole difficulty here.  These hosts carry, in
# `hostname -I` order: one 192.168.N.x per RDMA rail (NOT routable between
# nodes -- each rail is its own L2 segment), the routable management address,
# and assorted docker/libvirt bridge addresses.  getent on the short hostname
# returns the 127.0.1.1 /etc/hosts entry.  So: take the source address the
# kernel would use for off-box traffic, which is the management interface,
# then prove it before relying on it.
MASTER_IP=$(ip -4 route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+')
if [[ -z "$MASTER_IP" ]]; then
  echo "FATAL: could not determine a routable address for $MASTER_NODE"
  echo "candidates were: $(hostname -I)"; exit 1
fi

echo "=== topology ==="
echo "master node : $MASTER_NODE ($MASTER_IP)"
echo "peer node   : $PEER_NODE"
echo "grpc        : $MASTER_IP:$GRPC_PORT"
echo "master addrs: $(hostname -I)"
echo

# Prove reachability before spending the run on it: a silent connect failure
# here is indistinguishable downstream from "the feature does not work".
python3 -c "
import socket
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(('0.0.0.0',$GRPC_PORT)); s.listen(1)
c,_=s.accept(); c.close()
" &
PROBE_PID=$!
sleep 2
if ! srun --nodes=1 --ntasks=1 --nodelist="$PEER_NODE" \
     python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(8)
try:
    s.connect(('$MASTER_IP',$GRPC_PORT)); print('reachability OK')
except Exception as e:
    print('reachability FAILED:',e); sys.exit(1)
"; then
  echo "FATAL: peer node cannot reach $MASTER_IP:$GRPC_PORT"
  echo "master candidates were: $(hostname -I)"
  kill $PROBE_PID 2>/dev/null
  exit 1
fi
kill $PROBE_PID 2>/dev/null; wait $PROBE_PID 2>/dev/null
sleep 1

# TTL 10s / divisor 2 -> derived interval 5000ms; expiry window 10s x 3 = 30s.
export UMBP_HEARTBEAT_TTL_SEC=10
export UMBP_HEARTBEAT_INTERVAL_DIVISOR=2

cleanup() {
  [[ -n "${PEER_PID:-}" ]] && kill "$PEER_PID" 2>/dev/null
  [[ -n "${MASTER_PID:-}" ]] && kill "$MASTER_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

"$B/src/umbp/umbp_master" "0.0.0.0:$GRPC_PORT" "$METRICS_PORT" > /tmp/umbp_master_$SLURM_JOB_ID.log 2>&1 &
MASTER_PID=$!
sleep 5

if ! kill -0 "$MASTER_PID" 2>/dev/null; then
  echo "FATAL: master died on startup"; cat /tmp/umbp_master_$SLURM_JOB_ID.log; exit 1
fi

# The peer must reach the master across the network, so it gets the routable
# address rather than loopback.
# stdbuf: the peer is long-lived, so without line buffering its log stays empty
# in the failure case that most needs reading.
srun --nodes=1 --ntasks=1 --nodelist="$PEER_NODE" \
     --export=ALL \
     stdbuf -oL -eL \
     "$B/src/umbp/umbp_client" "$MASTER_IP:$GRPC_PORT" hb-peer-1 "$PEER_NODE:8080" \
     > /tmp/umbp_peer_$SLURM_JOB_ID.log 2>&1 &
PEER_PID=$!
sleep 12

# Heartbeat count for the remote peer, as recorded by the master.
hb() {
  curl -s "localhost:$METRICS_PORT/metrics" \
    | grep 'rpc_latency_seconds_count{node="hb-peer-1",rpc="Heartbeat"' \
    | awk '{print $2}'
}

if [[ -z "$(hb)" ]]; then
  echo "FATAL: no heartbeat metric for the peer"
  echo "--- is the master process alive? ---"
  kill -0 "$MASTER_PID" 2>/dev/null && echo "yes (pid $MASTER_PID)" || echo "NO"
  echo "--- is the peer step alive? ---"
  kill -0 "$PEER_PID" 2>/dev/null && echo "yes (pid $PEER_PID)" || echo "NO"
  echo "--- metrics endpoint says ---"
  curl -s --max-time 5 "localhost:$METRICS_PORT/metrics" | head -15
  echo "--- client_count ---"
  curl -s --max-time 5 "localhost:$METRICS_PORT/metrics" | grep -E 'client_count|node=' | head -5
  echo "--- master log ---"; tail -20 /tmp/umbp_master_$SLURM_JOB_ID.log
  echo "--- peer log ---";   tail -20 /tmp/umbp_peer_$SLURM_JOB_ID.log
  exit 1
fi

# Counts heartbeats over `window` seconds and reports the implied interval.
measure() {
  local label=$1 window=$2 expect_ms=$3
  local a b n
  a=$(hb); sleep "$window"; b=$(hb)
  n=$((b - a))
  if (( n > 0 )); then
    echo "$label: $n heartbeats in ${window}s -> ~$((window * 1000 / n))ms interval (expected ~${expect_ms}ms)"
  else
    echo "$label: 0 heartbeats in ${window}s (expected ~${expect_ms}ms) -- FAIL"
  fi
}

echo "=== phase 1: baseline (derived 10s/2) ==="
measure "baseline " 20 5000
echo

echo "=== phase 2: live change to 1000ms, no restart ==="
"$B/src/umbp/umbp_admin" "localhost:$GRPC_PORT" --heartbeat-interval-ms=1000
sleep 6   # converges within one old (5s) period
measure "after-set" 20 1000
echo

echo "=== phase 3: clamp guard (request 99999ms, expiry window is 30000ms) ==="
"$B/src/umbp/umbp_admin" "localhost:$GRPC_PORT" --heartbeat-interval-ms=99999
echo

echo "=== phase 4: clear the override, back to derived ==="
"$B/src/umbp/umbp_admin" "localhost:$GRPC_PORT" --heartbeat-interval-ms=0
sleep 6
measure "reverted " 20 5000
echo

echo "=== peer liveness (must be 1 throughout; a reaped peer would read 0) ==="
curl -s "localhost:$METRICS_PORT/metrics" | grep '^mori_umbp_client_count'
echo
echo "=== peer-side interval adoption log ==="
grep -i "Heartbeat interval updated" /tmp/umbp_peer_$SLURM_JOB_ID.log | head -5 \
  || echo "(peer log has no adoption line; peer logging may be silent by default)"
echo
echo "=== done ==="
