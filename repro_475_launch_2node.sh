#!/bin/bash
# Two-node launcher for the ROCm/mori#475 InterNodeV1 combine-corruption probe.
#
#   ./repro_475_launch_2node.sh                              # T=16..512, PASS/FAIL
#   COMBINE_IDX=orig ./repro_475_launch_2node.sh             # pass the original idx
#   SDMA=0 ./repro_475_launch_2node.sh                       # with SDMA off
#
# Run this ON rank 0's host; it drives rank 8..15 on $NODE1 over ssh. Defaults are
# wired for mi355-gpu-49 (rank 0) + mi355-gpu-51 (rank 1), container mori_zqz on
# podman, repo on shared NFS. Every value below is overridable from the env.
#
# Getting RDMA_N0/RDMA_N1 wrong makes the run hang in shmem_torch_process_group_init
# with no error message, which reads as "cannot reproduce" rather than
# "misconfigured" -- see the comment there.
set -u

NODE1=${NODE1:-mi355-gpu-51}             # rank 8..15; rank 0..7 run locally
PORT=${PORT:-29533}
CONTAINER=${CONTAINER:-mori_zqz}
CTL=${CTL:-podman}                       # container CLI, e.g. CTL='sudo -n docker'
IFACE=${IFACE:-enp193s0f1np1}            # management NIC, not the RDMA fabric
# Both nodes mount the repo from the same NFS export, so deriving it from this
# script's own location keeps the two sides in step with no second path to edit.
REPO=${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
LOGDIR=${LOGDIR:-$REPO/_logs475}         # must be visible from BOTH nodes
# rank 0 binds here and $NODE1 dials in, so it has to be this host's address on
# the management NIC rather than anything loopback-ish.
MASTER=${MASTER:-$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}
if [ -z "$MASTER" ]; then
  echo "cannot read an IPv4 address off IFACE=$IFACE; set MASTER= explicitly" >&2
  exit 2
fi
NPROC=${NPROC:-8}
DEADLINE=${DEADLINE:-900}
TAG=${TAG:-run}
SDMA=${SDMA:-1}
extra=("$@")

SCRIPT=${SCRIPT:-$REPO/repro_475_combine_probe.py}
GREP=${GREP:-'GLOBAL|RESULT|world=|kernel='}
# torchrun would otherwise report a missing script as eight identical per-rank
# exit code 2s, with the real message buried in the node log.
if [ ! -f "$SCRIPT" ]; then
  echo "probe script $SCRIPT does not exist" >&2; exit 2
fi

# mori pairs peers by device *index* into this list, so the two sides must list
# their NICs in the same rail order or every rank gets a peer on a different rail
# and no QP can form. These hosts name RDMA devices after PCI address and have
# identical layouts, so one sorted list serves both. It also has to exclude
# rocep193s0f{0,1} -- that is the management NIC ($IFACE), not fabric.
# Regenerate for other hosts by matching `rdma link` netdev names on both:
#     rdma link | awk '{print $2, $NF}'
RAILS=rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0,rocep25s0,rocep9s0
RDMA_N0=${RDMA_N0:-$RAILS}
RDMA_N1=${RDMA_N1:-$RAILS}

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
# COMBINE_IDX=orig makes the probe hand combine() the original routing indices
# instead of dispatch()'s returned ones -- the #475 root cause. Only tells the two
# apart on a mori without the dispatch_combine.py fix.
[ -n "${COMBINE_IDX:-}" ] && ENVS+=(-e COMBINE_IDX="$COMBINE_IDX")
# Set MORI_SRC=1 to run the in-tree build ($REPO/python) instead of the wheel in
# site-packages. $REPO is on shared storage, so both nodes pick up the same edits
# and the JIT compiles straight from $REPO/src -- no per-node copying.
[ -n "${MORI_SRC:-}" ] && ENVS+=(-e PYTHONPATH="$REPO/python")

torchrun_cmd() {  # $1 = node_rank -> prints the shell-quoted command
  printf '%q ' torchrun --nnodes=2 --nproc_per_node=$NPROC --node_rank="$1" \
                        --master_addr=$MASTER --master_port=$PORT \
                        "$SCRIPT" "${extra[@]}"
}

# Concurrent first-use JIT from many ranks into a shared cache dir can wedge
# (issue #475 latent bug 5). Build once per node, serially, before launching.
# Kept on one line: ssh re-splits the remote command on whitespace, so a
# multi-line -c payload arrives mangled and the node1 prebuild silently no-ops.
precompile="python -c 'from mori.jit.core import compile_genco; compile_genco(\"shmem_kernels\"); compile_genco(\"ep_internode_v1\")'"
$CTL exec "${ENVS[@]}" -w "$REPO" "$CONTAINER" bash -c "$precompile" >/dev/null 2>&1
ssh -o BatchMode=yes "$NODE1" \
  "$CTL exec $(printf '%q ' "${ENVS[@]}") -w $REPO $CONTAINER bash -c $(printf '%q' "$precompile")" \
  >/dev/null 2>&1

# node1 detached with output redirected *inside* the container: a foreground
# `ssh ... docker exec` had its container torn down when the ssh session ended,
# killing the run mid-sweep.
ssh -o BatchMode=yes "$NODE1" \
  $CTL exec -d "${ENVS[@]}" -e MORI_RDMA_DEVICES=$RDMA_N1 \
  -w "$REPO" "$CONTAINER" \
  bash -c "\"$(torchrun_cmd 1) > $N1 2>&1\""

timeout "$DEADLINE" $CTL exec "${ENVS[@]}" -e MORI_RDMA_DEVICES=$RDMA_N0 \
  -w "$REPO" "$CONTAINER" \
  bash -c "$(torchrun_cmd 0)" > "$N0" 2>&1
rc=$?

echo "=== $TAG (sdma=$SDMA combine_idx=${COMBINE_IDX:-dispatch} node0 rc=$rc) ==="
grep -hE "$GREP" "$N0"
if [ $rc -ne 0 ]; then
  echo "--- node0 errors ---"; grep -hE "error\]|Error|Traceback" "$N0" | head -8
  echo "(rc=124 means the run hit DEADLINE=${DEADLINE}s -- usually a QP-setup hang;"
  echo " check RDMA_N0/RDMA_N1 rail alignment and MORI_SOCKET_IFNAME)"
fi
echo "--- logs: $N0 $N1 ---"
exit 0
