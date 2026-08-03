#!/bin/bash
# Two-node launcher for the ROCm/mori#475 InterNodeV1 combine-corruption probes.
#
#   ./repro_475_launch_2node.sh                              # combine probe, T=16..512
#   PROBE=path ./repro_475_launch_2node.sh --rounds 4 --tokens 512   # path attribution
#   SDMA=0 ./repro_475_launch_2node.sh                       # with SDMA off
#
# Defaults are wired for the dev setup this was reproduced on (smci355-ccs-aus-n08-21
# as rank 0 + n06-21 as rank 1, container mori_zqz, /apps on shared NFS). Override
# via the env vars below.
#
# Getting RDMA_N0/RDMA_N1 wrong makes the run hang in shmem_torch_process_group_init
# with no error message, which reads as "cannot reproduce" rather than
# "misconfigured" -- see the comment there.
set -u

NODE1=${NODE1:-smci355-ccs-aus-n06-21}   # rank 8..15; rank 0..7 run locally
MASTER=${MASTER:-10.235.192.81}          # must be reachable from NODE1
PORT=${PORT:-29533}
CONTAINER=${CONTAINER:-mori_zqz}
IFACE=${IFACE:-enp81s0f1}                # management NIC, not the RDMA fabric
REPO=${REPO:-/apps/qizzhang/mori}
LOGDIR=${LOGDIR:-/apps/qizzhang/logs475} # must be visible from BOTH nodes
NPROC=${NPROC:-8}
DEADLINE=${DEADLINE:-900}
PROBE=${PROBE:-combine}                  # combine | path
TAG=${TAG:-run}
SDMA=${SDMA:-1}
extra=("$@")

case "$PROBE" in
  combine) SCRIPT=$REPO/repro_475_combine_probe.py;     GREP='GLOBAL|RESULT|world=|kernel=' ;;
  path)    SCRIPT=$REPO/repro_475_path_attribution.py;  GREP='^\[path\]' ;;
  *) echo "PROBE must be 'combine' or 'path', got '$PROBE'" >&2; exit 2 ;;
esac

# The two nodes enumerate ionic_N against different rails, and mori pairs peers by
# device *index*. Left alone that pairs each rank with a NIC on a different rail,
# which cannot form a QP. MORI_RDMA_DEVICES preserves the listed order, so listing
# each node's devices in rail1..rail8 order aligns the two sides. Regenerate for
# other hosts by matching `rdma link` netdev names on both:
#     rdma link | awk '{print $2, $NF}'
RDMA_N0=${RDMA_N0:-ionic_0,ionic_1,ionic_3,ionic_2,ionic_4,ionic_5,ionic_7,ionic_6}
RDMA_N1=${RDMA_N1:-ionic_2,ionic_3,ionic_1,ionic_7,ionic_5,ionic_4,ionic_6,ionic_0}

mkdir -p "$LOGDIR"
N0=$LOGDIR/n0_$TAG.log
N1=$LOGDIR/n1_$TAG.log
: > "$N0"; : > "$N1"

ENVS=(
  -e MORI_SHMEM_HEAP_SIZE=8G
  -e MORI_DISABLE_AUTO_XGMI=0
  -e MORI_ENABLE_SDMA=$SDMA
  -e GLOO_SOCKET_IFNAME=$IFACE
  -e NCCL_SOCKET_IFNAME=$IFACE
  # The backend fabric is rail-optimized, so a NCCL ring pairing arbitrary ranks
  # would cross rails and hang. Only mori's own QPs should touch the NICs.
  -e NCCL_IB_DISABLE=1
  # mori's bootstrap ring is a socket ring of its own, separate from gloo/nccl.
  # Without this it can pick an unroutable interface (docker0/tunl0) and the run
  # dies after a 300 s PhoneHomeProtocol timeout.
  -e MORI_SOCKET_IFNAME=$IFACE
  # Unrelated to #475, but required to run at all on a pair whose AINIC firmware
  # versions differ: CCQE is auto-detected per node, so the two sides would build
  # different JIT kernels and QP setup would never complete. Pinning it just makes
  # both sides agree; drop this on a uniform pair if you want the default.
  -e MORI_DISABLE_IONIC_CCQE=1
  -e MORI_GLOBAL_LOG_LEVEL=warn
)
[ -n "${PATH_CLASSES:-}" ] && ENVS+=(-e PATH_CLASSES="$PATH_CLASSES")

torchrun_cmd() {  # $1 = node_rank -> prints the shell-quoted command
  printf '%q ' torchrun --nnodes=2 --nproc_per_node=$NPROC --node_rank="$1" \
                        --master_addr=$MASTER --master_port=$PORT \
                        "$SCRIPT" "${extra[@]}"
}

# Concurrent first-use JIT from many ranks into a shared cache dir can wedge
# (issue #475 latent bug 5). Build once per node, serially, before launching.
for pre in "" "ssh -o BatchMode=yes $NODE1"; do
  $pre sudo -n docker exec "${ENVS[@]}" -w "$REPO" "$CONTAINER" \
    python -c "from mori.jit.core import compile_genco
compile_genco('shmem_kernels')
compile_genco('ep_internode_v1')" >/dev/null 2>&1
done

# node1 detached with output redirected *inside* the container: a foreground
# `ssh ... docker exec` had its container torn down when the ssh session ended,
# killing the run mid-sweep.
ssh -o BatchMode=yes "$NODE1" \
  sudo -n docker exec -d "${ENVS[@]}" -e MORI_RDMA_DEVICES=$RDMA_N1 \
  -w "$REPO" "$CONTAINER" \
  bash -c "\"$(torchrun_cmd 1) > $N1 2>&1\""

timeout "$DEADLINE" sudo -n docker exec "${ENVS[@]}" -e MORI_RDMA_DEVICES=$RDMA_N0 \
  -w "$REPO" "$CONTAINER" \
  bash -c "$(torchrun_cmd 0)" > "$N0" 2>&1
rc=$?

echo "=== $TAG (probe=$PROBE sdma=$SDMA node0 rc=$rc) ==="
grep -hE "$GREP" "$N0"
if [ $rc -ne 0 ]; then
  echo "--- node0 errors ---"; grep -hE "error\]|Error|Traceback" "$N0" | head -8
  echo "(rc=124 means the run hit DEADLINE=${DEADLINE}s -- usually a QP-setup hang;"
  echo " check RDMA_N0/RDMA_N1 rail alignment and MORI_SOCKET_IFNAME)"
fi
echo "--- logs: $N0 $N1 ---"
exit 0
